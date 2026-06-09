import SwiftUI

// ── API Response ──────────────────────────────────────────────────────────────

struct AnalyzeResponse: Codable, Equatable {
    let imageId:             String
    let processedImageSize:  [Int]        // [w, h] at backend processing resolution
    let summary:             String
    let objects:             [DetectedObject]
    let guide:               Guide
    let annotatedImageUrl:   String
    let latencyMs:           Int

    enum CodingKeys: String, CodingKey {
        case imageId            = "image_id"
        case processedImageSize = "processed_image_size"
        case summary, objects, guide
        case annotatedImageUrl  = "annotated_image_url"
        case latencyMs          = "latency_ms"
    }

    var processedWidth:  Int { processedImageSize.first ?? 1 }
    var processedHeight: Int { processedImageSize.last  ?? 1 }
}

struct DetectedObject: Codable, Identifiable, Equatable {
    let id:          Int
    let label:       String
    let description: String?   // nil when describe=OFF
    let state:       String?   // trash_signal | needs_wash | tidy_target | in_use | nil
    let category:    String
    let categoryKo:  String
    let decision:    String?   // "discard" | "relocate" | "keep" | nil (no API key)
    let confidence:  Float
    let bbox:        [Int]     // [x1, y1, x2, y2] in processedImageSize space

    enum CodingKeys: String, CodingKey {
        case id, label, description, state, category
        case categoryKo = "category_ko"
        case decision, confidence, bbox
    }

    // Decision display helpers (match orchestrator.py _DECISION_COLORS)
    var decisionColor: Color {
        switch decision {
        case "discard":  return .red
        case "relocate": return .orange
        case "keep":     return Color(red: 0, green: 0.78, blue: 0.31)
        default:         return .gray
        }
    }

    var decisionKo: String {
        switch decision {
        case "discard":  return "버릴 것"
        case "relocate": return "옮길 것"
        case "keep":     return "제자리"
        default:         return "미정"
        }
    }
}

struct Guide: Codable, Equatable {
    let priorityTop3:    [String]
    let fiveMinRoutine:  [String]
    let byCategory:      [String: String]

    enum CodingKeys: String, CodingKey {
        case priorityTop3   = "priority_top3"
        case fiveMinRoutine = "five_min_routine"
        case byCategory     = "by_category"
    }
}
