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
}

// MARK: - AnalysisResult

/// HomeView의 sheet 표시를 위한 결과 묶음. Identifiable이므로 sheet(item:) 바인딩에 사용 가능.
struct AnalysisResult: Identifiable {
    let id = UUID()
    let image: UIImage
    let response: AnalyzeResponse
}
