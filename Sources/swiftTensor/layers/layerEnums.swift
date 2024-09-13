public enum padStyle: Sendable {
    case explicit, valid, same
}

public enum padMode: Sendable {
    case zero, reflect, symmetric, clamp, const
}

public enum convDataLayout: Sendable {
    case NHWC, NCHW
}

public enum convWeightLayout: Sendable {
    case OIHW, HWIO
}
