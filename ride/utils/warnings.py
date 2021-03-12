# Filter out library warnings
import warnings

from torch.jit import TracerWarning


def filter_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The pts_unit 'pts' gives wrong results and will be removed in a follow-up version. Please use pts_unit 'sec'.",
    )
    warnings.filterwarnings(
        "ignore",
        message="Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.",
    )
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!",
    )
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!",
    )
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!",
    )
    warnings.filterwarnings(
        "ignore",
        message="Integer division of tensors using div or / is deprecated, and in a future release div will perform true division as in Python 3. Use true_divide or floor_divide (// in Python) instead.",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Iterating over a tensor might cause the trace to be incorrect.[\w: ()]*",
    )
    warnings.filterwarnings(
        "ignore", message=r"Skipped operation [\w:]* [\d]+ time\(s\)"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=TracerWarning)
