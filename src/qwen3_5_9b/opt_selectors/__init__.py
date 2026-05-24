"""Selector registry for the implemented sentence-level methods.

The release code exposes ILP, MMR, and DPP selectors under
``opt_selectors/sentence_level``.  Summary-level selectors are not part of the
current paper-table run path.
"""


def _lazy_import(method):
    """延迟导入选择器函数，避免在 import 阶段就加载所有依赖库。"""
    if method == "ilp":
        from opt_selectors.sentence_level.ilp import ilp_select
        return ilp_select
    elif method == "mmr":
        from opt_selectors.sentence_level.mmr import mmr_select
        return mmr_select
    elif method == "dpp":
        from opt_selectors.sentence_level.dpp import dpp_select
        return dpp_select
    else:
        raise ValueError(f"Unknown selector: {method}")


SENTENCE_LEVEL_METHODS = {"ilp", "mmr", "dpp"}
ALL_METHODS = SENTENCE_LEVEL_METHODS


def get_selector(name):
    """根据方法名返回对应的选择器函数（延迟导入）。"""
    if name not in ALL_METHODS:
        raise ValueError(f"Unknown selector: {name}. Available: {sorted(ALL_METHODS)}")
    return _lazy_import(name)
