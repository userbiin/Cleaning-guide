import UIKit

// MARK: - Mock API Service

/// 네트워크 없이 Preview와 테스트에서 사용하는 목 서비스
final class MockAPIService: AnalysisService {
    var result: Result<AnalyzeResponse, Error> = .success(PreviewData.sampleResponse)
    var delay: Duration = .zero

    func analyze(image: UIImage, describe: Bool, detector: String) async throws -> AnalyzeResponse {
        if delay > .zero { try await Task.sleep(for: delay) }
        return try result.get()
    }
}

// MARK: - Preview Sample Data

/// 모든 #Preview 블록에서 공유하는 정적 목 데이터
enum PreviewData {

    // MARK: Sample Image

    /// 합성 UIImage — 960×720, bbox 위치에 색상 블록을 그려 overlay 미리보기 지원
    static let sampleImage: UIImage = {
        let size = CGSize(width: 960, height: 720)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { ctx in
            UIColor.systemGray6.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))

            // bottle (id 0) bbox: [100, 80, 300, 380]
            UIColor.systemRed.withAlphaComponent(0.25).setFill()
            ctx.fill(CGRect(x: 100, y: 80, width: 200, height: 300))

            // book (id 1) bbox: [400, 200, 650, 380]
            UIColor.systemOrange.withAlphaComponent(0.25).setFill()
            ctx.fill(CGRect(x: 400, y: 200, width: 250, height: 180))

            // cable (id 2) bbox: [700, 100, 800, 150]
            UIColor.systemGreen.withAlphaComponent(0.25).setFill()
            ctx.fill(CGRect(x: 700, y: 100, width: 100, height: 50))

            // tissue (id 3) bbox: [50, 500, 180, 580]
            UIColor.systemGray.withAlphaComponent(0.25).setFill()
            ctx.fill(CGRect(x: 50, y: 500, width: 130, height: 80))
        }
    }()

    // MARK: Sample Objects

    /// 4개 객체 — 모든 decision 상태(discard/relocate/keep/nil)와 describe ON/OFF 경로 커버
    static let sampleObjects: [DetectedObject] = [
        DetectedObject(
            id: 0,
            label: "bottle",
            description: "A crumpled empty plastic bottle",
            state: "trash_signal",
            category: "kitchen_items",
            categoryKo: "식기/음식 관련",
            decision: "discard",
            confidence: 0.92,
            bbox: [100, 80, 300, 380]
        ),
        DetectedObject(
            id: 1,
            label: "book",
            description: "A thick textbook lying open",
            state: nil,
            category: "study_supplies",
            categoryKo: "학습/문구류",
            decision: "relocate",
            confidence: 0.87,
            bbox: [400, 200, 650, 380]
        ),
        DetectedObject(
            id: 2,
            label: "cable",
            description: nil,
            state: nil,
            category: "electronics",
            categoryKo: "전자기기/케이블",
            decision: "keep",
            confidence: 0.63,
            bbox: [700, 100, 800, 150]
        ),
        DetectedObject(
            id: 3,
            label: "tissue",
            description: nil,
            state: "trash_signal",
            category: "daily_supplies",
            categoryKo: "생활용품",
            decision: nil,
            confidence: 0.41,
            bbox: [50, 500, 180, 580]
        ),
    ]

    // MARK: Sample Guide

    static let sampleGuide = Guide(
        priorityTop3: [
            "빈 페트병을 재활용 분리수거함에 버리세요.",
            "교재를 책꽂이에 꽂아두세요.",
            "케이블을 케이블 정리함에 모으세요."
        ],
        fiveMinRoutine: [
            "바닥에 있는 쓰레기를 모두 줍기",
            "책상 위 음료/식기를 주방으로 옮기기",
            "케이블을 한 곳에 모으기",
            "책과 노트를 책꽂이에 정리하기",
            "빈 공간을 물티슈로 닦기"
        ],
        byCategory: [
            "식기/음식 관련": "컵과 식기는 주방으로 옮기고, 음식 포장지는 바로 버리세요.",
            "학습/문구류": "책은 책꽂이에 세우고, 종이는 파일철에 모으세요."
        ]
    )

    // MARK: Sample Response

    /// processedImageSize [960, 720] = sampleImage 크기 → BBoxOverlayView 좌표 1:1 매핑
    static let sampleResponse = AnalyzeResponse(
        imageId: "preview_001",
        processedImageSize: [960, 720],
        summary: "책상 위에 물건이 4개 감지되었습니다. 페트병은 버리고, 책은 책꽂이로 옮기세요.",
        objects: sampleObjects,
        guide: sampleGuide,
        annotatedImageUrl: "",
        latencyMs: 1234
    )
}
