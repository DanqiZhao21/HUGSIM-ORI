from __future__ import annotations


def normalize_ad_name(ad: str) -> str:
    aliases = {
        "uniad": "uniad",
        "vad": "vad",
        "ltf": "ltf",
        "sparsedrive-v2": "sparsedrive_v2",
        "sparsedrive_v2": "sparsedrive_v2",
    }

    normalized = aliases.get(ad)
    if normalized is None:
        raise NotImplementedError(f"Unsupported AD backend: {ad}")

    return normalized


def resolve_ad_launch_env(cfg, ad: str) -> dict[str, str]:
    normalized = normalize_ad_name(ad)

    if normalized == "uniad":
        env = {}
        if hasattr(cfg.base, "uniad_repo"):
            env["UNIAD_REPO"] = cfg.base.uniad_repo
        if hasattr(cfg.base, "uniad_conda_env"):
            env["UNIAD_CONDA_ENV"] = cfg.base.uniad_conda_env
        if hasattr(cfg.base, "uniad_checkpoint"):
            env["UNIAD_CHECKPOINT"] = cfg.base.uniad_checkpoint
        if hasattr(cfg.base, "uniad_config"):
            env["UNIAD_CONFIG"] = cfg.base.uniad_config
        return env
    if normalized == "sparsedrive_v2":
        return {
            "SPARSEDRIVE_PYTHON_BIN": cfg.base.sparsedrive_v2_python,
            "SPARSEDRIVE_CHECKPOINT": cfg.base.sparsedrive_v2_ckpt,
            "NUPLAN_DEVKIT_ROOT": cfg.base.sparsedrive_v2_nuplan_repo,
            "HUGSIM_SCENE_NAME": cfg.scenario.scene_name,
            "HUGSIM_MODEL_BASE": cfg.base.model_base,
            "HUGSIM_DT": str(cfg.kinematic.dt),
            "HUGSIM_WHEELBASE": str(cfg.kinematic.Lf + cfg.kinematic.Lr),
        }
    if normalized in {"vad", "ltf"}:
        return {}

    raise NotImplementedError(f"Unsupported AD backend: {ad}")


def resolve_ad_path(cfg, ad: str) -> str:
    normalized = normalize_ad_name(ad)

    if normalized == "uniad":
        return cfg.base.uniad_path
    if normalized == "vad":
        return cfg.base.vad_path
    if normalized == "ltf":
        return cfg.base.ltf_path
    if normalized == "sparsedrive_v2":
        return cfg.base.sparsedrive_v2_path

    raise NotImplementedError(f"Unsupported AD backend: {ad}")
