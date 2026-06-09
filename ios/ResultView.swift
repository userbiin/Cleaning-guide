import SwiftUI

struct ResultView: View {
    let image:    UIImage
    let response: AnalyzeResponse

    @State private var selectedID: Int?

    private var selectedObject: DetectedObject? {
        response.objects.first { $0.id == selectedID }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                imageWithOverlay
                contentSection
            }
            .padding(.bottom, 40)
        }
        .navigationTitle("분석 결과")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Image + BBox

    private var imageWithOverlay: some View {
        // .overlay on Image gives GeometryReader exactly the rendered image size,
        // which correctly handles aspect-ratio scaling (scaledToFit).
        Image(uiImage: image)
            .resizable()
            .scaledToFit()
            .overlay(alignment: .topLeading) {
                GeometryReader { geo in
                    BBoxOverlayView(
                        objects:       response.objects,
                        displaySize:   geo.size,
                        processedSize: CGSize(
                            width:  CGFloat(response.processedWidth),
                            height: CGFloat(response.processedHeight)
                        ),
                        selectedID: $selectedID
                    )
                }
            }
    }

    // MARK: - Content below image

    private var contentSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Summary
            if !response.summary.isEmpty {
                Text(response.summary)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            }

            // Decision colour legend
            HStack(spacing: 16) {
                ForEach(legendItems, id: \.label) { item in
                    HStack(spacing: 5) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(item.color)
                            .frame(width: 12, height: 12)
                        Text(item.label)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal)

            // Object chips + detail card
            if !response.objects.isEmpty {
                objectSection
            }

            // Guide
            if hasGuide {
                GuideView(guide: response.guide)
                    .padding(.horizontal)
            }

            // Latency
            Text("처리 시간: \(response.latencyMs) ms")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.horizontal)
        }
    }

    private var hasGuide: Bool {
        !response.guide.priorityTop3.isEmpty ||
        !response.guide.fiveMinRoutine.isEmpty ||
        !response.guide.byCategory.isEmpty
    }

    private let legendItems: [(label: String, color: Color)] = [
        ("버릴 것", .red),
        ("옮길 것", .orange),
        ("제자리",  Color(red: 0, green: 0.78, blue: 0.31)),
        ("미정",   .gray),
    ]

    // MARK: - Object section

    private var objectSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("탐지된 물건 (\(response.objects.count)개)")
                .font(.headline)
                .padding(.horizontal)

            // Horizontal chip strip
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(response.objects) { obj in
                        ObjectChip(object: obj, isSelected: selectedID == obj.id)
                            .onTapGesture {
                                withAnimation(.spring(duration: 0.25)) {
                                    selectedID = selectedID == obj.id ? nil : obj.id
                                }
                            }
                    }
                }
                .padding(.horizontal)
            }

            // Detail card for tapped object
            if let obj = selectedObject {
                ObjectDetailCard(object: obj)
                    .padding(.horizontal)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .animation(.spring(duration: 0.25), value: selectedID)
            }
        }
    }
}

// MARK: - Object chips

private struct ObjectChip: View {
    let object:     DetectedObject
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(object.decisionColor)
                .frame(width: 7, height: 7)
            Text(object.label)
                .font(.caption.weight(.medium))
            Text(String(format: "%.0f%%", object.confidence * 100))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            isSelected ? object.decisionColor.opacity(0.12) : Color(.tertiarySystemBackground),
            in: Capsule()
        )
        .overlay(Capsule().strokeBorder(isSelected ? object.decisionColor : .clear, lineWidth: 1.5))
    }
}

private struct ObjectDetailCard: View {
    let object: DetectedObject

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(object.label)
                        .font(.headline)
                    Text(object.categoryKo)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                DecisionBadge(decision: object.decision)
            }

            // Florence-2 description (describe=ON path only)
            if let desc = object.description, !desc.isEmpty {
                Text("\u{201C}\(desc)\u{201D}")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .italic()
            }

            // Parsed state signal from attributes.py
            if let state = object.state {
                Label(state, systemImage: "wand.and.sparkles")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Decision badge

struct DecisionBadge: View {
    let decision: String?

    private var info: (String, Color) {
        switch decision {
        case "discard":  return ("버릴 것", .red)
        case "relocate": return ("옮길 것", .orange)
        case "keep":     return ("제자리",  Color(red: 0, green: 0.78, blue: 0.31))
        default:         return ("미정",   .gray)
        }
    }

    var body: some View {
        let (label, color) = info
        Text(label)
            .font(.caption.bold())
            .foregroundStyle(color)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(color.opacity(0.15), in: Capsule())
    }
}

// MARK: - Guide view

private struct GuideView: View {
    let guide: Guide

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("정리 가이드")
                .font(.headline)

            if !guide.priorityTop3.isEmpty {
                GuideSection(title: "우선순위 TOP 3", icon: "exclamationmark.triangle.fill", color: .orange) {
                    ForEach(Array(guide.priorityTop3.enumerated()), id: \.offset) { i, item in
                        HStack(alignment: .top, spacing: 10) {
                            Text("\(i + 1)")
                                .font(.caption.bold())
                                .foregroundStyle(.white)
                                .frame(width: 20, height: 20)
                                .background(.orange, in: Circle())
                            Text(item).font(.subheadline)
                        }
                    }
                }
            }

            if !guide.fiveMinRoutine.isEmpty {
                GuideSection(title: "5분 정리 루틴", icon: "timer", color: .blue) {
                    ForEach(Array(guide.fiveMinRoutine.enumerated()), id: \.offset) { i, step in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: "\(i + 1).circle.fill").foregroundStyle(.blue)
                            Text(step).font(.subheadline)
                        }
                    }
                }
            }

            if !guide.byCategory.isEmpty {
                GuideSection(title: "카테고리별 가이드", icon: "tag.fill", color: .purple) {
                    ForEach(Array(guide.byCategory.sorted(by: { $0.key < $1.key })), id: \.key) { cat, advice in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(cat).font(.caption.bold()).foregroundStyle(.purple)
                            Text(advice).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }
}

private struct GuideSection<Content: View>: View {
    let title:   String
    let icon:    String
    let color:   Color
    let content: () -> Content

    init(title: String, icon: String, color: Color, @ViewBuilder content: @escaping () -> Content) {
        self.title   = title
        self.icon    = icon
        self.color   = color
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(color)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Previews

#Preview("전체 결과") {
    NavigationStack {
        ResultView(image: PreviewData.sampleImage, response: PreviewData.sampleResponse)
    }
}

#Preview("가이드 없음") {
    let noGuideResponse = AnalyzeResponse(
        imageId: "preview_002",
        processedImageSize: [960, 720],
        summary: "간단한 분석 결과입니다.",
        objects: Array(PreviewData.sampleObjects.prefix(2)),
        guide: Guide(priorityTop3: [], fiveMinRoutine: [], byCategory: [:]),
        annotatedImageUrl: "",
        latencyMs: 500
    )
    return NavigationStack {
        ResultView(image: PreviewData.sampleImage, response: noGuideResponse)
    }
}

#Preview("DecisionBadge - 버릴 것") {
    DecisionBadge(decision: "discard").padding()
}

#Preview("DecisionBadge - 제자리") {
    DecisionBadge(decision: "keep").padding()
}

#Preview("DecisionBadge - 미정") {
    DecisionBadge(decision: nil).padding()
}
