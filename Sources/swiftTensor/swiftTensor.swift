import Foundation
import Metal
import MetalPerformanceShaders




public enum dataType {
    case float16, float32, bfloat16, float8
    case int8, int16, int32
    case uint8, uint16, uint32
    case qint4, qint2, quint4, quint2
}

public enum computeType {
    case metal, cpu, accelerate, cuda, mps
}

public enum tensorOperations {
    case noOP
    case add
}

//public struct CPUStorage<S: CPUDatatType> {
//    var gradient: Array<S>?
//    var data: Array<S>?
//}

//public struct MetalStorage {
//    let device: MTLDevice
//    var gradient: MTLBuffer?
//    var data: MTLBuffer?
//    var commandQue: MTLCommandQueue
//    var commandBuffer: MPSCommandBuffer?
//    var commandEncoder: MTLComputeCommandEncoder?
//    var heap: MTLHeap
//}
//

public protocol TenosrType: Hashable {
    associatedtype StorageType
}

//public struct CPU<S: CPUDatatType>: TenosrType {
//    public typealias StorageType = CPUStorage<S>
//}
//
//public struct Metal: TenosrType {
//    public typealias StorageType = MetalStorage
//}

public final class Tensor<T: TenosrType>: Hashable {
    public var storage: T.StorageType
    public let shape: [Int]
    public let dataType: dataType
    public var op: tensorOperations
    public var requiresGradient: Bool
    public var name: String? = nil
    public var childrens: [Tensor]? = nil
    
    public init(storage: T.StorageType, shape: [Int], dataType: dataType, requiresGradient: Bool = false, childrens: [Tensor]?, op: tensorOperations = .noOP, name: String? = nil) {
        self.storage = storage
        self.shape = shape
        self.dataType = dataType
        self.requiresGradient = requiresGradient
        self.childrens = childrens
        self.name = name
        self.op = op
    }
}


public extension Tensor {
    static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

public extension Tensor {
    @inline(__always)
    func checkBinary(_ right: Tensor) {
        precondition(self.dataType == right.dataType, "dtype didnt match")
//        precondition(self.storage.data != nil && right.data != nil, "data didnt exists")
    }
}


//public extension CPUStorage<Float32> {
//    init(data: inout [Float32]) {
//        data.withUnsafeMutableBufferPointer {
//            self.data = $0
//        }
//    }
//}
