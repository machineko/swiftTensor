public enum padStyle {
    case explicit, valid, same
}

public enum padMode {
    case zero, reflect, symmetric, clamp, const
}

public enum convDataLayout {
    case NHWC, NCHW
}

public enum convWeightLayout {
    case OIHW, HWIO
}