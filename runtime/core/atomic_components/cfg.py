import os
import pickle


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)
    

def parse_cfg(cfg_pkl, data_root, replace_cfg=None):

    def _check_path(p):
        if os.path.isfile(p):
            return p
        else:
            return os.path.join(data_root, p)

    cfg = load_pkl(cfg_pkl)

    if isinstance(replace_cfg, dict):
        for k, v in replace_cfg.items():
            if not isinstance(v, dict):
                continue
            for kk, vv in v.items():
                cfg[k][kk] = vv

    # Force CPU mode when CUDA is unavailable or explicitly disabled
    _force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
    _cuda_disabled = not os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if _force_cpu or _cuda_disabled:
        def _set_device_cpu(obj):
            if isinstance(obj, dict):
                if "device" in obj and isinstance(obj["device"], str) and obj["device"].startswith("cuda"):
                    obj["device"] = "cpu"
                for v in obj.values():
                    _set_device_cpu(v)
            elif isinstance(obj, list):
                for item in obj:
                    _set_device_cpu(item)
        _set_device_cpu(cfg)

    base_cfg = cfg["base_cfg"]
    audio2motion_cfg = cfg["audio2motion_cfg"]
    default_kwargs = cfg["default_kwargs"]

    for k in base_cfg:
        if k == "landmark478_cfg":
            for kk in ["task_path", "blaze_face_model_path", "face_mesh_model_path"]:
                if kk in base_cfg[k] and base_cfg[k][kk]:
                    base_cfg[k][kk] = _check_path(base_cfg[k][kk])
        else:
            base_cfg[k]["model_path"] = _check_path(base_cfg[k]["model_path"])

    audio2motion_cfg["model_path"] = _check_path(audio2motion_cfg["model_path"])

    avatar_registrar_cfg = {
        k: base_cfg[k]
        for k in [
            "insightface_det_cfg",
            "landmark106_cfg",
            "landmark203_cfg",
            "landmark478_cfg",
            "appearance_extractor_cfg",
            "motion_extractor_cfg",
        ]
    }

    stitch_network_cfg = base_cfg["stitch_network_cfg"]
    warp_network_cfg = base_cfg["warp_network_cfg"]
    decoder_cfg = base_cfg["decoder_cfg"]
    
    condition_handler_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "use_emo",
            "use_sc",
            "use_eye_open",
            "use_eye_ball",
            "seq_frames",
        ]
    }

    lmdm_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "model_path",
            "device",
            "motion_feat_dim",
            "audio_feat_dim",
            "seq_frames",
        ]
    }

    wav2feat_cfg = {"w2f_cfg": base_cfg["hubert_cfg"]}
    
    return [
        avatar_registrar_cfg,
        condition_handler_cfg,
        lmdm_cfg,
        stitch_network_cfg,
        warp_network_cfg,
        decoder_cfg,
        wav2feat_cfg,
        default_kwargs,
    ]
