import Testing
@testable import swiftTensor
import Foundation

extension UnsafeMutableRawPointer {
    func toArray<T>(count: Int) -> [T] {
        return Array(UnsafeBufferPointer(start: self.assumingMemoryBound(to: T.self), count: count))
    }
}

extension Optional where Wrapped == UnsafeMutableRawPointer {
    func toArray<T>(count: Int) -> [T]? {
        return self?.toArray(count: count)
    }
}


@Test func example() async throws {
    var aData: [Float32] = [1,2]
    var bData: [Float32] = [2,3]
    let apointer = aData.withUnsafeMutableBytes { UnsafeMutableRawPointer($0.baseAddress!) }
    let bpointer = bData.withUnsafeMutableBytes { UnsafeMutableRawPointer($0.baseAddress!) }

    var a = Tensor(shape: [2], childrens: [], data: apointer)
    var b = Tensor(shape: [2], childrens: [], data: bpointer)
    
    let c = await Tensor(fromChildrens: &a, &b, shape: [2])
    print(c.childrens)
    print(tensorStorage)
    let zz = tensorStorage[c.childrens[0]]
    print(zz?.pointee.shape)
//    zz?.pointee
    let d1: [Float32] = a.data!.toArray(count: 2)
    zz?.pointee.data = bpointer
    let d2: [Float32] = a.data!.toArray(count: 2)
    print(d1, d2)
}
