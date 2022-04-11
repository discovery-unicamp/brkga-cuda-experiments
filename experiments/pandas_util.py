from typing import Any, Iterable, Tuple
import pandas as pd


def tuples(df: pd.DataFrame) -> Iterable[Tuple[Any, ...]]:
    return df.itertuples(index=False, name=None)
