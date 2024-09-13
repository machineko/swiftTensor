import Foundation
#if canImport(MetalPerformanceShadersGraph)
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
#endif



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
    case equal, notEqual
    case power, maximum, minimum, modulo
    case floor, rounded
    case matMul
    case reshape, permute, select, transpose
    case slice
    case gather, expandDim
    case mean, variance
    case sqrt
    case conv2d
    case coordinate
    case tile, broadcast
    case attention
    case concatenate
    case reciprocal
    case clamp
    case max, min
    case resizeNearest, resizeBilinear
    case softmax, sigmoid, tanh, gelu

}


public protocol CPUDataType: AdditiveArithmetic {}

extension Float16: CPUDataType {}
extension Float32: CPUDataType {}
extension Float64: CPUDataType {}
extension Int32: CPUDataType {}

public final class CPUStorage {
    public var gradient: UnsafeMutableRawBufferPointer?
    public var data: UnsafeMutableRawBufferPointer?

    public init(data: UnsafeMutableRawBufferPointer, gradient: UnsafeMutableRawBufferPointer? = nil) {
        self.data = data
        self.gradient = gradient
    }

    deinit {
        data?.deallocate()
        gradient?.deallocate()
    }

    // Convenience initializer for UnsafeMutableBufferPointer
    public convenience init<T>(data: UnsafeMutableBufferPointer<T>) where T: CPUDataType {
        let rawBuffer = UnsafeMutableRawBufferPointer(data)
        self.init(data: rawBuffer)
    }

    // Convenience initializer for Array
    public convenience init<T>(_ array: [T]) where T: CPUDataType {
        let count = array.count
        let byteCount = count * MemoryLayout<T>.stride
        let alignment = MemoryLayout<T>.alignment
        let rawBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: alignment)

        array.withUnsafeBytes { bufferPointer in
            rawBuffer.copyMemory(from: UnsafeRawBufferPointer(bufferPointer))
        }

        self.init(data: rawBuffer)
    }
}

public struct CPU: TensorType {
    public typealias StorageType = CPUStorage
}
extension Tensor {
    public var dataByteSize: Int {
        return self.numberOfElements * self.dataType.byteSize
    }

    @inline(__always)
    public func checkBinary(_ right: Tensor) where T.StorageType == CPUStorage {
        precondition(self.dataType == right.dataType, "dtype didnt match")
        precondition(self.storage.data != nil && right.storage.data != nil, "data didnt exists")
    }
}
#if canImport(MetalPerformanceShadersGraph)

public struct MPGTensor: TensorType {
    public typealias StorageType = MPGraphStorage
}


public final class MPGraphStorage {
    public var data: MPSGraphTensor?
    public var gradient: MPSGraphTensor?
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public var commandBuffer: MPSCommandBuffer?
    public var commandEncoder: MTLComputeCommandEncoder?
    public var exec: MPSGraphExecutable?
    public let serialQue: DispatchQueue?
    public var heap: MTLHeap?


    public init(
        data: MPSGraphTensor? = nil,
        gradient: MPSGraphTensor? = nil,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        commandBuffer: MPSCommandBuffer? = nil,
        commandEncoder: MTLComputeCommandEncoder? = nil,
        exec: MPSGraphExecutable? = nil,
        serialQue: DispatchQueue? = nil,
        heap: MTLHeap? = nil
    ) {
        self.data = data
        self.gradient = gradient
        self.device = device
        self.commandQueue = commandQueue
        self.commandBuffer = commandBuffer
        self.commandEncoder = commandEncoder
        self.exec = exec
        self.serialQue = serialQue
        self.heap = heap
    }

    public convenience init(
        data: MPSGraphTensor,
        gradient: MPSGraphTensor? = nil,
        parent: MPGraphStorage
    ) {
        self.init(
            data: data, gradient: gradient, device: parent.device, commandQueue: parent.commandQueue, 
            commandBuffer: parent.commandBuffer, commandEncoder: parent.commandEncoder, 
            exec: parent.exec, serialQue: parent.serialQue, heap: parent.heap)
    }

}
#endif

public extension UnsafeMutableRawBufferPointer {
    func toArray<T>(of type: T.Type) -> [T] where T: CPUDataType {
        precondition(self.count % MemoryLayout<T>.stride == 0, "Buffer size is not a multiple of the element size")
        let count = self.count / MemoryLayout<T>.stride
        return Array(UnsafeBufferPointer(start: self.baseAddress?.assumingMemoryBound(to: T.self), count: count))
    }
}

public enum RoundingRule {
    case toNearestOrAwayFromZero  // equivalent to .rounded()
    case towardZero  // equivalent to .rounded(.towardZero)
    case awayFromZero
    case toNearestOrEven
    case up  // equivalent to .rounded(.up)
    case down  // equivalent to .rounded(.down)
}

public protocol TensorType {
    associatedtype StorageType
}

public protocol ConvStorage {
    associatedtype StorageType
}

public protocol ShapeType {
    var shape: [Int] { get }
    init(_ shape: [Int])
}

public struct Shape1D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 1, "1 Dimmension supported, passed = \(shape.count)")
        self.shape = shape
    }
}

public struct Shape2D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 2, "2 Dimmension supported, passed = \(shape.count)")
        self.shape = shape
    }
}

public struct Shape3D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 3, "3 Dimmension supported, passed = \(shape.count)")
        self.shape = shape
    }
}

public struct Shape4D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 4, "4 Dimmension supported, passed = \(shape.count)")
        self.shape = shape
    }
}

public struct Shape5D: ShapeType {
    public let shape: [Int]
    public init(_ shape: [Int]) {
        precondition(shape.count == 5, "5 Dimmension supported, passed = \(shape.count)")
        self.shape = shape
    }
}

public typealias oneDim = Shape1D
public typealias twoDim = Shape2D
public typealias threeDim = Shape3D
public typealias fourDim = Shape4D
public typealias fiveDim = Shape5D

public enum AnyTensor {
    case cpu(Tensor<CPU, oneDim>)
    case cpu2D(Tensor<CPU, twoDim>)
    case cpu3D(Tensor<CPU, threeDim>)
    case cpu4D(Tensor<CPU, fourDim>)
    case cpu5D(Tensor<CPU, fiveDim>)
    #if canImport(MetalPerformanceShadersGraph)

    case mps(Tensor<MPGTensor, oneDim>)
    case mps2D(Tensor<MPGTensor, twoDim>)
    case mps3D(Tensor<MPGTensor, threeDim>)
    case mps4D(Tensor<MPGTensor, fourDim>)
    case mps5D(Tensor<MPGTensor, fiveDim>)
    #endif
    var shape: [Int] {
        switch self {
            case .cpu(let t): return t.shape.shape
            case .cpu2D(let t): return t.shape.shape
            case .cpu3D(let t): return t.shape.shape
            case .cpu4D(let t): return t.shape.shape
            case .cpu5D(let t): return t.shape.shape
            
            #if canImport(MetalPerformanceShadersGraph)
            case .mps(let t):return t.shape.shape
            case .mps2D(let t): return t.shape.shape
            case .mps3D(let t) :return t.shape.shape
            case .mps4D(let t): return t.shape.shape
            case .mps5D(let t): return t.shape.shape
            #endif

            }
    }

    var dataType: dataType {
        switch self {
        case .cpu(let t1D): return t1D.dataType
        case .cpu2D(let t2D): return t2D.dataType
        case .cpu3D(let t3D): return t3D.dataType
        case .cpu4D(let t4D): return t4D.dataType
        case .cpu5D(let t5D): return t5D.dataType
        #if canImport(MetalPerformanceShadersGraph)

        case .mps(let t1D): return t1D.dataType
        case .mps2D(let t2D): return t2D.dataType
        case .mps3D(let t3D): return t3D.dataType
        case .mps4D(let t4D): return t4D.dataType
        case .mps5D(let t5D): return t5D.dataType
        #endif
        }
    }

    var isCPU: Bool {
        switch self {
        case .cpu, .cpu2D, .cpu3D, .cpu4D, .cpu5D: return true
        default: return false
        }
    }
    #if canImport(MPSGraph)
    var isMPSGraph: Bool {
        switch self {
        case .mps, .mps2D, .mps3D, .mps4D, .mps5D: return true
        default: return false
        }
    }
    #endif
}



public final class Tensor<T: TensorType, S: ShapeType>: Hashable {
    public var storage: T.StorageType
    public let shape: S
    public let dataType: dataType
    public var op: tensorOperations
    public var requiresGradient: Bool
    public var name: String? = nil
    public var childrens: [AnyTensor]? = nil
    
    public init(storage: T.StorageType, shape: S, dataType: dataType, requiresGradient: Bool = false, childrens: [AnyTensor]?, op: tensorOperations, name: String? = nil) {
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

public extension oneDim {
    init(_ array: [any Numeric]) {
        self.shape = [array.count]
    }
}

public extension Tensor {
    func wrapInEnum() -> AnyTensor {
        switch (T.self, S.self) {
        case is (CPU.Type, oneDim.Type): return .cpu(self as! Tensor<CPU, oneDim>)
        case is (CPU.Type, twoDim.Type): return .cpu2D(self as! Tensor<CPU, twoDim>)
        case is (CPU.Type, threeDim.Type): return .cpu3D(self as! Tensor<CPU, threeDim>)
        case is (CPU.Type, fourDim.Type): return .cpu4D(self as! Tensor<CPU, fourDim>)
        case is (CPU.Type, fiveDim.Type): return .cpu5D(self as! Tensor<CPU, fiveDim>)
        #if canImport(MetalPerformanceShadersGraph)
        case is (MPGTensor.Type, oneDim.Type): return .mps(self as! Tensor<MPGTensor, oneDim>)
        case is (MPGTensor.Type, twoDim.Type): return .mps2D(self as! Tensor<MPGTensor, twoDim>)
        case is (MPGTensor.Type, threeDim.Type): return .mps3D(self as! Tensor<MPGTensor, threeDim>)
        case is (MPGTensor.Type, fourDim.Type): return .mps4D(self as! Tensor<MPGTensor, fourDim>)
        case is (MPGTensor.Type, fiveDim.Type): return .mps5D(self as! Tensor<MPGTensor, fiveDim>)
        #endif
        default: fatalError("Unsupported tensor type, \(T.self), \(S.self)")
        }
    }
}

public extension ShapeType {
    var count: Int {
        return self.shape.count
    }
}
