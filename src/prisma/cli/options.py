from typing import Any, TypeVar, Callable

import click

from .utils import PathlibPath

FC = TypeVar('FC', bound=click.Command | Callable[..., Any])


schema: Callable[[FC], FC] = click.option(  # pyright: ignore[reportGeneralTypeIssues]
    '--schema',
    type=PathlibPath(exists=True, dir_okay=False, resolve_path=True),
    help='The location of the Prisma schema file.',
    required=False,
)

watch: Callable[[FC], FC] = click.option(  # pyright: ignore[reportGeneralTypeIssues]
    '--watch',
    is_flag=True,
    default=False,
    required=False,
    help='Watch the Prisma schema and rerun after a change',
)

skip_generate: Callable[[FC], FC] = click.option(  # pyright: ignore[reportGeneralTypeIssues]
    '--skip-generate',
    is_flag=True,
    default=False,
    help='Skip triggering generators (e.g. Prisma Client Python)',
)
