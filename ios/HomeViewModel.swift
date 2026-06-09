import SwiftUI
import PhotosUI
import Observation

// MARK: - HomeViewModel

@Observable
final class HomeViewModel {

    // MARK: State

    var selectedImage: UIImage?
    var isAnalyzing = false
    /// non-nil → ResultView sheet 표시
    var analysisResult: AnalysisResult?
    var errorMessage: String?
    var serverStatus: ServerStatus = .unknown

    // MARK: Computed

    var canAnalyze: Bool { selectedImage != nil && !isAnalyzing }

    // MARK: Dependency

    /// 기본값: 실제 APIService 생성. Preview/테스트 시 MockAPIService 주입 가능.
    private let serviceFactory: (String) -> AnalysisService

    init(serviceFactory: @escaping (String) -> AnalysisService = { APIService(baseURL: $0) }) {
        self.serviceFactory = serviceFactory
    }

    // MARK: Actions

    /// PhotosPickerItem에서 UIImage를 비동기로 로드
    func loadImage(from item: PhotosPickerItem?) async {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let img = UIImage(data: data) else { return }
        selectedImage = img
    }

    /// 서버에 이미지 분석 요청. @AppStorage 값을 파라미터로 받아 VM이 SwiftUI에 의존하지 않게 함.
    func analyze(serverURL: String, useDescribe: Bool) async {
        guard let image = selectedImage else { return }
        isAnalyzing = true
        defer { isAnalyzing = false }

        let service = serviceFactory(serverURL)
        do {
            let response = try await service.analyze(
                image: image,
                describe: useDescribe,
                detector: "yolo-world"
            )
            analysisResult = AnalysisResult(image: image, response: response)
        } catch let urlErr as URLError {
            errorMessage = "네트워크 오류 (\(urlErr.code.rawValue)). 서버가 실행 중인지 확인하세요."
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// GET /health 를 호출해 서버 연결 상태를 확인.
    func checkHealth(serverURL: String) async {
        var base = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while base.hasSuffix("/") { base = String(base.dropLast()) }
        guard !base.isEmpty, let url = URL(string: "\(base)/health") else {
            serverStatus = .failed("잘못된 URL")
            return
        }
        serverStatus = .checking
        do {
            let req = URLRequest(url: url, timeoutInterval: 8)
            let (_, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                serverStatus = .ok
            } else {
                serverStatus = .failed("서버 오류")
            }
        } catch {
            serverStatus = .failed("연결 실패")
        }
    }

    // MARK: - Server Status

    enum ServerStatus {
        case unknown, checking, ok, failed(String)

        var dotColor: Color {
            switch self {
            case .ok:      return .green
            case .failed:  return .red
            case .checking: return .orange
            case .unknown: return Color(.systemGray3)
            }
        }

        var label: String {
            switch self {
            case .ok:            return "연결됨"
            case .failed(let m): return m
            case .checking:      return "확인 중…"
            case .unknown:       return "미확인"
            }
        }
    }
}

// MARK: - AnalysisResult

/// HomeView의 sheet 표시를 위한 결과 묶음. Identifiable이므로 sheet(item:) 바인딩에 사용 가능.
struct AnalysisResult: Identifiable {
    let id = UUID()
    let image: UIImage
    let response: AnalyzeResponse
}
