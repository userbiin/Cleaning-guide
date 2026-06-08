import Foundation
import UIKit

enum APIError: LocalizedError {
    case invalidImage
    case server(statusCode: Int, message: String?)
    case decoding

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "이미지를 처리할 수 없습니다."
        case .server(let code, let msg):
            return msg ?? "서버 오류 (\(code)). 잠시 후 다시 시도하세요."
        case .decoding:
            return "응답을 파싱하지 못했습니다."
        }
    }
}

final class APIService {
    private let baseURL: String
    private let session: URLSession

    init(baseURL: String) {
        var url = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while url.hasSuffix("/") { url = String(url.dropLast()) }
        self.baseURL = url

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        session = URLSession(configuration: config)
    }

    func analyze(image: UIImage, describe: Bool, detector: String = "yolo-world") async throws -> AnalyzeResponse {
        guard let url = URL(string: "\(baseURL)/analyze") else {
            throw URLError(.badURL)
        }
        // Normalize EXIF orientation to .up before encoding.
        // The backend applies its own EXIF correction; sending a pre-normalized
        // image (orientation tag = 1) makes that step a no-op.
        guard let jpeg = normalizedJPEG(image, quality: 0.85) else {
            throw APIError.invalidImage
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        req.httpBody = multipartBody(boundary: boundary, jpeg: jpeg, describe: describe, detector: detector)

        let (data, response) = try await session.data(for: req)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let serverMsg = (try? JSONDecoder().decode([String: String].self, from: data))?["error"]
            throw APIError.server(statusCode: http.statusCode, message: serverMsg)
        }

        do {
            return try JSONDecoder().decode(AnalyzeResponse.self, from: data)
        } catch {
            throw APIError.decoding
        }
    }

    private func normalizedJPEG(_ image: UIImage, quality: CGFloat) -> Data? {
        guard image.imageOrientation != .up else {
            return image.jpegData(compressionQuality: quality)
        }
        let renderer = UIGraphicsImageRenderer(size: image.size)
        let normalized = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
        return normalized.jpegData(compressionQuality: quality)
    }

    private func multipartBody(boundary: String, jpeg: Data, describe: Bool, detector: String) -> Data {
        var body = Data()
        body.appendFilePart(name: "image", filename: "photo.jpg", contentType: "image/jpeg", data: jpeg, boundary: boundary)
        body.appendTextPart(name: "describe", value: describe ? "true" : "false", boundary: boundary)
        body.appendTextPart(name: "detector",  value: detector, boundary: boundary)
        body.append("--\(boundary)--\r\n")
        return body
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let d = string.data(using: .utf8) { append(d) }
    }

    mutating func appendTextPart(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        append(value)
        append("\r\n")
    }

    mutating func appendFilePart(name: String, filename: String, contentType: String, data: Data, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(contentType)\r\n\r\n")
        append(data)
        append("\r\n")
    }
}
