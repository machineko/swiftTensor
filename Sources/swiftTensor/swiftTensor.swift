import Foundation
import Metal
import MetalPerformanceShaders




public enum dataType {
    case float16, float32, bfloat16, float8, float64
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
    // case qint4, qint2, quint4, quint2
}

public enum computeType {
    case metal, cpu, accelerate, cuda, mps
}

public enum tensorOperations {
    case noOP
    case add, subtract, multiply, divide
    case power, maximum, minimum, modulo
    case floor, rounded
    case matMul
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

// public protocol CPUDatatType {}

// extension Float32: CPUDatatType {}
// extension Float64: CPUDatatType {}
// extension Int32: CPUDatatType {}


// public struct CPUStorage<S: CPUDatatType> {
//    var gradient: Array<S>?
//    var data: Array<S>?
// }

// public struct CPU<S: CPUDatatType>: TenosrType {
//    public typealias StorageType = CPUStorage<S>
// }




//public struct CPU<S: CPUDatatType>: TenosrType {
//    public typealias StorageType = CPUStorage<S>
//}
//
//public struct Metal: TenosrType {
//    public typealias StorageType = MetalStorage
//}
public protocol TensorType {
    associatedtype StorageType
}

public protocol ShapeType {
    var shape: [Int] { get }
}

public struct Shape1D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 1, "1 Dimmension supported")
        self.shape = shape
    }
}

public struct Shape2D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 2, "2 Dimmension supported")
        self.shape = shape
    }
}

public struct Shape3D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 3, "3 Dimmension supported")
        self.shape = shape
    }
}

public typealias oneDim = Shape1D
public typealias twoDim = Shape2D
public typealias threeDim = Shape3D

public final class Tensor<T: TensorType, S: ShapeType>: Hashable {
    public var storage: T.StorageType
    public let shape: S
    public let dataType: dataType
    public var op: tensorOperations
    public var requiresGradient: Bool
    public var name: String? = nil
    public var childrens: [Tensor]? = nil
    
    public init(storage: T.StorageType, shape: S, dataType: dataType, requiresGradient: Bool = false, childrens: [Tensor]?, op: tensorOperations = .noOP, name: String? = nil) {
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
    var numberOfElements: Int {
        self.shape.shape.reduce(1, *)
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


public extension dataType {
    var byteSize: Int {
        switch self {
        case .float16:
            return 2
        case .float32:
            return 4
        case .bfloat16:
            return 2
        case .float8:
            return 1
        case .int8:
            return 1
        case .int16:
            return 2
        case .int32:
            return 4
        case .uint8:
            return 1
        case .uint16:
            return 2
        case .uint32:
            return 4
        case .float64:
            return 8
        case .int64:
            return 8
        case .uint64:
            return 8

}
    }
}


//public extension CPUStorage<Float32> {
//    init(data: inout [Float32]) {
//        data.withUnsafeMutableBufferPointer {
//            self.data = $0
//        }
//    }
//}

public extension oneDim {
    init(_ array: [any Numeric]) {
        self.shape = [array.count]
    }
}
