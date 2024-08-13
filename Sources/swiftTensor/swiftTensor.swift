import Foundation

protocol CPUDatatTypes {}

extension Float32: CPUDatatTypes {}
@available(macOS 11.0, *)
extension Float16: CPUDatatTypes {}

public enum dataType {
    case float16, float32, bfloat16, float8
    case int8, int16, int32
    case uint8, uint16, uint32
    case qint4, qint2, quint4, quint2
}

public enum backendType {
    case metal, cpu, accelerate, cuda, mps
}

public enum tensorOperations {
    case id
    case add
}

protocol tensorStorage: Hashable {
    associatedtype Storage
}

public struct CPU: tensorStorage {
    typealias Storage = UnsafeMutableBufferPointer<CPUDatatTypes>
}

final class Tensor<S: tensorStorage>: Hashable {
    let backend: backendType
    var data: S.Storage? = nil
    var gradient: S.Storage? = nil
    let shape: [Int]
    let dataType: dataType
    var op: tensorOperations
    var requiresGradient: Bool
    var name: String? = nil
    var childrens: [Tensor]
    
    public init(backend: backendType, gradient: S.Storage, shape: [Int], dataType: dataType, requiresGradient: Bool, childrens: [Tensor], op: tensorOperations = .id, name: String? = nil) {
        self.backend = backend
        self.gradient = gradient
        self.shape = shape
        self.dataType = dataType
        self.requiresGradient = requiresGradient
        self.childrens = childrens
        self.name = name
        self.op = .id
    }
}

extension Tensor {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}
