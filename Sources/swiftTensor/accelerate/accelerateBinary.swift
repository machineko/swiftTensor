#if os(macOS)
import Accelerate

extension Tensor<CPU<Float32>> {
    static func +(_ l: Tensor, _ r: Tensor) -> Tensor {
        l.checkBinary(r)
        guard l.dataType == .float32 else {
            fatalError("Only f32 supported")
        }
        var out = UnsafeMutableBufferPointer<Float>.allocate(capacity: l.shape.reduce(1, *))
        vDSP.add(l.storage.data!, r.storage.data!, result: &out)
        let storage = CPUStorage(data: out)
        return Tensor<CPU<Float32>>(storage: storage, shape: l.shape, dataType: l.dataType, requiresGradient: l.requiresGradient, childrens: [l, r], op: .add)
    }

}
#endif
