import os
import sys
import json
import shutil
import logging
import subprocess
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Generic, Optional, cast
from pathlib import Path
from contextvars import ContextVar
from typing_extensions import override

from jinja2 import Environment, StrictUndefined, FileSystemLoader
from pydantic import BaseModel, ValidationError

from . import jsonrpc
from .. import __version__
from .types import PartialModel
from .utils import (
    copy_tree,
    is_same_path,
    resolve_template_path,
)
from ..utils import DEBUG, DEBUG_GENERATOR
from .errors import PartialTypeGeneratorError
from .models import PythonData, DefaultData
from .._types import BaseModelT, InheritsGeneric, get_args
from .filters import quote
from .jsonrpc import Manifest
from .._compat import model_json, model_parse, cached_property

__all__ = (
    'BASE_PACKAGE_DIR',
    'GenericGenerator',
    'BaseGenerator',
    'Generator',
    'render_template',
    'cleanup_templates',
    'partial_models_ctx',
)

log: logging.Logger = logging.getLogger(__name__)
BASE_PACKAGE_DIR = Path(__file__).parent.parent
GENERIC_GENERATOR_NAME = 'prisma.generator.generator.GenericGenerator'

# set of templates that should be rendered after every other template
DEFERRED_TEMPLATES = {'partials.py.jinja'}

# templates that require special handling (not rendered in the normal loop)
# All types/ templates are handled by _render_model_types to ensure proper cleanup
SPECIAL_TEMPLATES = {
    'types/_model.py.jinja',
    'types/__init__.py.jinja',
    'types/filters.py.jinja',
    'types/atomic.py.jinja',
    'types/list_filters.py.jinja',
}

DEFAULT_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    loader=FileSystemLoader(Path(__file__).parent / 'templates'),
    undefined=StrictUndefined,
)

# the type: ignore is required because Jinja2 filters are not typed
# and Pyright infers the type from the default builtin filters which
# results in an overly restrictive type
DEFAULT_ENV.filters['quote'] = quote  # pyright: ignore

partial_models_ctx: ContextVar[List[PartialModel]] = ContextVar('partial_models_ctx', default=[])  # noqa: B039


class GenericGenerator(ABC, Generic[BaseModelT]):
    @abstractmethod
    def get_manifest(self) -> Manifest:
        """Get the metadata for this generator

        This is used by prisma to display the post-generate message e.g.

        ✔ Generated Prisma Client Python to ./.venv/lib/python3.10/site-packages/prisma
        """
        ...

    @abstractmethod
    def generate(self, data: BaseModelT) -> None: ...

    @classmethod
    def invoke(cls) -> None:
        """Shorthand for calling BaseGenerator().run()"""
        generator = cls()
        generator.run()

    def run(self) -> None:
        """Run the generation loop

        This can only be called from a prisma generation, e.g.

        ```prisma
        generator client {
            provider = "python generator.py"
        }
        ```
        """
        if not os.environ.get('PRISMA_GENERATOR_INVOCATION'):
            raise RuntimeError('Attempted to invoke a generator outside of Prisma generation')

        request = None
        try:
            while True:
                line = jsonrpc.readline()
                if line is None:
                    log.debug('Prisma invocation ending')
                    break

                request = jsonrpc.parse(line)
                self._on_request(request)
        except Exception as exc:
            if request is None:
                raise exc from None

            # We don't care about being overly verbose or printing potentially redundant data here
            if DEBUG or DEBUG_GENERATOR:
                traceback.print_exc()

            # Do not include the full stack trace for pydantic validation errors as they are typically
            # the fault of the user.
            if isinstance(exc, ValidationError):
                message = str(exc)
            elif isinstance(exc, PartialTypeGeneratorError):
                # TODO: remove our internal frame from this stack trace
                message = (
                    'An exception ocurred while running the partial type generator\n' + traceback.format_exc().strip()
                )
            else:
                message = traceback.format_exc().strip()

            response = jsonrpc.ErrorResponse(
                id=request.id,
                error={
                    # code copied from https://github.com/prisma/prisma/blob/main/packages/generator-helper/src/generatorHandler.ts
                    'code': -32000,
                    'message': message,
                    'data': {},
                },
            )
            jsonrpc.reply(response)

    def _on_request(self, request: jsonrpc.Request) -> None:
        response = None
        if request.method == 'getManifest':
            response = jsonrpc.SuccessResponse(
                id=request.id,
                result=dict(
                    manifest=self.get_manifest(),
                ),
            )
        elif request.method == 'generate':
            if request.params is None:  # pragma: no cover
                raise RuntimeError('Prisma JSONRPC did not send data to generate.')

            if DEBUG_GENERATOR:
                _write_debug_data('params', json.dumps(request.params, indent=2))

            data = model_parse(self.data_class, request.params)

            if DEBUG_GENERATOR:
                _write_debug_data('data', model_json(data, indent=2))

            self.generate(data)
            response = jsonrpc.SuccessResponse(id=request.id, result=None)
        else:  # pragma: no cover
            raise RuntimeError(f'JSON RPC received unexpected method: {request.method}')

        jsonrpc.reply(response)

    @cached_property
    def data_class(self) -> Type[BaseModelT]:
        """Return the BaseModel used to parse the Prisma DMMF"""

        # we need to cast to object as otherwise pyright correctly marks the code as unreachable,
        # this is because __orig_bases__ is not present in the typeshed stubs as it is
        # intended to be for internal use only, however I could not find a method
        # for resolving generic TypeVars for inherited subclasses without using it.
        # please create an issue or pull request if you know of a solution.
        cls = cast(object, self.__class__)
        if not isinstance(cls, InheritsGeneric):
            raise RuntimeError('Could not resolve generic type arguments.')

        typ: Optional[Any] = None
        for base in cls.__orig_bases__:
            if base.__origin__ == GenericGenerator:
                typ = base
                break

        if typ is None:  # pragma: no cover
            raise RuntimeError(
                'Could not find the GenericGenerator type;\n'
                'This should never happen;\n'
                f'Does {cls} inherit from {GenericGenerator} ?'
            )

        args = get_args(typ)
        if not args:
            raise RuntimeError(f'Could not resolve generic arguments from type: {typ}')

        model = args[0]
        if not issubclass(model, BaseModel):
            raise TypeError(
                f'Expected first generic type argument argument to be a subclass of {BaseModel} '
                f'but got {model} instead.'
            )

        # we know the type we have resolved is the same as the first generic argument
        # passed to GenericGenerator, safe to cast
        return cast(Type[BaseModelT], model)


class BaseGenerator(GenericGenerator[DefaultData]):
    pass


class Generator(GenericGenerator[PythonData]):
    @override
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f'{Generator} cannot be subclassed, maybe you meant {BaseGenerator}?')

    @override
    def get_manifest(self) -> Manifest:
        return Manifest(
            name=f'Prisma Client Python (v{__version__})',
            default_output=BASE_PACKAGE_DIR,
            requires_engines=[
                'queryEngine',
            ],
        )

    @override
    def generate(self, data: PythonData) -> None:
        config = data.generator.config
        rootdir = Path(data.generator.output.value)
        if not rootdir.exists():
            rootdir.mkdir(parents=True, exist_ok=True)

        if not is_same_path(BASE_PACKAGE_DIR, rootdir):
            copy_tree(BASE_PACKAGE_DIR, rootdir)

        # copy the Prisma Schema file used to generate the client to the
        # package so we can use it to instantiate the query engine
        packaged_schema = rootdir / 'schema.prisma'
        if not is_same_path(data.schema_path, packaged_schema):
            packaged_schema.write_text(data.datamodel)

        params = data.to_params()

        try:
            for name in DEFAULT_ENV.list_templates():
                if not name.endswith('.py.jinja') or name.startswith('_') or name in DEFERRED_TEMPLATES:
                    continue

                # Skip templates that require special handling
                if name in SPECIAL_TEMPLATES:
                    continue

                # Skip templates in subdirectories that start with _ (e.g., types/_model.py.jinja)
                parts = name.split('/')
                if len(parts) > 1 and parts[-1].startswith('_'):
                    continue

                render_template(rootdir, name, params)

            # Generate per-model type files
            _render_model_types(rootdir, data, params)

            if config.partial_type_generator:
                log.debug('Generating partial types')
                config.partial_type_generator.run()

            params['partial_models'] = partial_models_ctx.get()
            for name in DEFERRED_TEMPLATES:
                render_template(rootdir, name, params)
        except:
            cleanup_templates(rootdir, env=DEFAULT_ENV)
            raise

        # Optionally run ruff to clean up unused imports
        _run_ruff_if_available(rootdir)

        log.debug('Finished generating Prisma Client Python')


def _run_ruff_if_available(rootdir: Path) -> None:
    """Run ruff to fix linting issues if available."""
    try:
        # Check if ruff is available
        result = subprocess.run(
            ['ruff', '--version'],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            log.debug('ruff not available, skipping linting fixes')
            return

        # Run ruff to fix various issues in the types directory:
        # - F401: unused imports
        # - I001: import sorting
        # - E501: line too long (ruff format handles this)
        types_dir = rootdir / 'types'
        if types_dir.exists():
            # First run ruff check --fix for auto-fixable lint issues
            subprocess.run(
                ['ruff', 'check', '--select', 'F401,I001', '--fix', str(types_dir)],
                capture_output=True,
                check=False,
            )
            # Then run ruff format to fix line length and formatting
            subprocess.run(
                ['ruff', 'format', str(types_dir)],
                capture_output=True,
                check=False,
            )
            log.debug('Ran ruff to fix linting issues in %s', types_dir)
    except FileNotFoundError:
        log.debug('ruff not found, skipping linting fixes')
    except Exception as e:
        log.debug('Failed to run ruff: %s', e)


def cleanup_templates(rootdir: Path, *, env: Optional[Environment] = None) -> None:
    """Revert module to pre-generation state"""
    if env is None:
        env = DEFAULT_ENV

    for name in env.list_templates():
        file = resolve_template_path(rootdir=rootdir, name=name)
        if file.exists():
            log.debug('Removing rendered template at %s', file)
            file.unlink()

    # Also clean up dynamically generated model type files
    types_dir = rootdir / 'types'
    if types_dir.exists():
        for file in types_dir.glob('*.py'):
            # Don't remove __init__.py as it's generated from a template
            if file.name != '__init__.py':
                log.debug('Removing generated model types at %s', file)
                file.unlink()


def _render_model_types(rootdir: Path, data: PythonData, params: Dict[str, Any]) -> None:
    """Render all type files in the types/ directory."""
    types_dir = rootdir / 'types'

    # Remove any existing types/ directory before generating
    # This prevents stale files from a previous schema from persisting
    if types_dir.exists():
        shutil.rmtree(types_dir)
        log.debug('Removed existing types directory at %s', types_dir)

    types_dir.mkdir(parents=True, exist_ok=True)

    # Render static type templates (filters, atomic, list_filters)
    for template_name in ('types/filters.py.jinja', 'types/atomic.py.jinja', 'types/list_filters.py.jinja'):
        render_template(rootdir, template_name, params)

    # Get the model template
    model_template = DEFAULT_ENV.get_template('types/_model.py.jinja')

    # Get all models for cross-references
    all_models = data.dmmf.datamodel.models

    # Render a file for each model
    for model in all_models:
        model_schema = params['type_schema'].get_model(model.name)
        model_params = {
            **params,
            'model': model,
            'model_schema': model_schema,
            'all_models': all_models,
        }

        output = model_template.render(**model_params)
        file = types_dir / f'{model.name.lower()}.py'
        file.write_bytes(output.encode(sys.getdefaultencoding()))
        log.debug('Rendered model types to %s', file.absolute())

    # Render the types/__init__.py.jinja template
    render_template(rootdir, 'types/__init__.py.jinja', params)


def render_template(
    rootdir: Path,
    name: str,
    params: Dict[str, Any],
    *,
    env: Optional[Environment] = None,
) -> None:
    if env is None:
        env = DEFAULT_ENV

    template = env.get_template(name)
    output = template.render(**params)

    file = resolve_template_path(rootdir=rootdir, name=name)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    file.write_bytes(output.encode(sys.getdefaultencoding()))
    log.debug('Rendered template to %s', file.absolute())


def _write_debug_data(name: str, output: str) -> None:
    path = Path(__file__).parent.joinpath(f'debug-{name}.json')

    with path.open('w') as file:
        file.write(output)

    log.debug('Wrote generator %s to %s', name, path.absolute())
