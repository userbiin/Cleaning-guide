import SwiftUI

// MARK: - Dark theme palette (shared with HomeView)
private extension Color {
    static let appBackground  = Color(red: 0.04, green: 0.04, blue: 0.06)
    static let cardBackground = Color(red: 0.11, green: 0.11, blue: 0.14)
    static let cardBorder     = Color.white.opacity(0.08)
    static let neonMint       = Color(red: 0.18, green: 0.95, blue: 0.75)
    static let labelPrimary   = Color.white
    static let labelSecondary = Color.white.opacity(0.55)
    static let labelTertiary  = Color.white.opacity(0.30)
}

struct ResultView: View {
    let image:    UIImage
    let response: AnalyzeResponse

    @State private var selectedID: Int?

    private var selectedObject: DetectedObject? {
        response.objects.first { $0.id == selectedID }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                imageWithOverlay
                    .padding(.bottom, 24)

                contentSection
                    .padding(.bottom, 48)
            }
        }
        .background(Color.appBackground)
        .preferredColorScheme(.dark)
        .navigationTitle("분석 결과")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Color.appBackground, for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
    }

    // MARK: - Image + BBox

    private var imageWithOverlay: some View {
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

    // MARK: - Content

    private var contentSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            if !response.summary.isEmpty {
                summaryCard
            }

            statsRow

            if !response.objects.isEmpty {
                objectSection
            }

            if hasGuide {
                GuideView(guide: response.guide)
                    .padding(.horizontal)
            }

            HStack(spacing: 5) {
                Image(systemName: "clock")
                    .font(.caption2)
                Text("처리 시간: \(response.latencyMs) ms")
                    .font(.caption2)
            }
            .foregroundStyle(Color.labelTertiary)
            .padding(.horizontal)
        }
    }

    private var summaryCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "text.bubble.fill")
                .font(.title3)
                .foregroundStyle(Color.neonMint)
                .padding(.top, 2)
            Text(response.summary)
                .font(.subheadline)
                .foregroundStyle(Color.labelPrimary)
                .lineSpacing(3)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.neonMint.opacity(0.07), in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(Color.neonMint.opacity(0.25), lineWidth: 1)
        )
        .padding(.horizontal)
    }

    private var statsRow: some View {
        HStack(spacing: 10) {
            ForEach(legendItems, id: \.label) { item in
                let count = response.objects.filter { $0.decisionColor == item.color }.count
                StatChip(label: item.label, color: item.color, count: count)
            }
            Spacer()
        }
        .padding(.horizontal)
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
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("탐지된 물건")
                    .font(.headline.bold())
                    .foregroundStyle(Color.labelPrimary)
                Text("\(response.objects.count)개")
                    .font(.subheadline)
                    .foregroundStyle(Color.labelSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.cardBackground, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.cardBorder, lineWidth: 1))
            }
            .padding(.horizontal)

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

            if let obj = selectedObject {
                ObjectDetailCard(object: obj)
                    .padding(.horizontal)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .animation(.spring(duration: 0.25), value: selectedID)
            }
        }
    }
}

// MARK: - Stat Chip

private struct StatChip: View {
    let label: String
    let color: Color
    let count: Int

    var body: some View {
        VStack(spacing: 2) {
            Text("\(count)")
                .font(.headline.bold())
                .foregroundStyle(count > 0 ? color : Color.white.opacity(0.2))
            Text(label)
                .font(.caption2)
                .foregroundStyle(Color.white.opacity(0.5))
        }
        .frame(minWidth: 52)
        .padding(.vertical, 10)
        .padding(.horizontal, 6)
        .background(count > 0 ? color.opacity(0.1) : Color(red: 0.11, green: 0.11, blue: 0.14), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(count > 0 ? color.opacity(0.3) : Color.white.opacity(0.06), lineWidth: 1)
        )
    }
}

// MARK: - Object Chip

private struct ObjectChip: View {
    let object:     DetectedObject
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(object.decisionColor)
                .frame(width: 7, height: 7)
            Text(object.label)
                .font(.caption.weight(.medium))
                .foregroundStyle(Color.white)
            Text(String(format: "%.0f%%", object.confidence * 100))
                .font(.caption2)
                .foregroundStyle(Color.white.opacity(0.45))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            isSelected ? object.decisionColor.opacity(0.18) : Color(red: 0.11, green: 0.11, blue: 0.14),
            in: Capsule()
        )
        .overlay(
            Capsule().strokeBorder(
                isSelected ? object.decisionColor.opacity(0.7) : Color.white.opacity(0.08),
                lineWidth: isSelected ? 1.5 : 1
            )
        )
        .shadow(color: isSelected ? object.decisionColor.opacity(0.3) : .clear, radius: 8, x: 0, y: 0)
        .scaleEffect(isSelected ? 1.04 : 1.0)
        .animation(.spring(duration: 0.2), value: isSelected)
    }
}

// MARK: - Object Detail Card

private struct ObjectDetailCard: View {
    let object: DetectedObject

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(object.label)
                        .font(.headline.bold())
                        .foregroundStyle(Color.white)
                    Text(object.categoryKo)
                        .font(.caption)
                        .foregroundStyle(Color.white.opacity(0.5))
                }
                Spacer()
                DecisionBadge(decision: object.decision)
            }

            if let desc = object.description, !desc.isEmpty {
                Divider().background(Color.white.opacity(0.08))
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "quote.opening")
                        .font(.caption2)
                        .foregroundStyle(Color.white.opacity(0.35))
                        .padding(.top, 2)
                    Text(desc)
                        .font(.caption)
                        .foregroundStyle(Color.white.opacity(0.55))
                        .italic()
                }
            }

            if let state = object.state {
                HStack(spacing: 5) {
                    Image(systemName: "wand.and.sparkles")
                        .font(.caption2)
                    Text(state)
                        .font(.caption2)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.orange.opacity(0.12), in: Capsule())
                .overlay(Capsule().strokeBorder(Color.orange.opacity(0.35), lineWidth: 1))
                .foregroundStyle(.orange)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [
                    object.decisionColor.opacity(0.1),
                    Color(red: 0.11, green: 0.11, blue: 0.14)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(object.decisionColor.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: object.decisionColor.opacity(0.12), radius: 12, x: 0, y: 4)
    }
}

// MARK: - Decision Badge

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
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(color.opacity(0.15), in: Capsule())
            .overlay(
                Capsule().strokeBorder(color.opacity(0.4), lineWidth: 1)
            )
    }
}

// MARK: - Guide View

private struct GuideView: View {
    let guide: Guide

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                Image(systemName: "list.clipboard.fill")
                    .foregroundStyle(Color(red: 0.18, green: 0.95, blue: 0.75))
                Text("정리 가이드")
                    .font(.headline.bold())
                    .foregroundStyle(Color.white)
            }

            if !guide.priorityTop3.isEmpty {
                GuideSection(title: "우선순위 TOP 3", icon: "exclamationmark.triangle.fill", color: .orange) {
                    ForEach(Array(guide.priorityTop3.enumerated()), id: \.offset) { i, item in
                        HStack(alignment: .top, spacing: 10) {
                            Text("\(i + 1)")
                                .font(.caption.bold())
                                .foregroundStyle(.black)
                                .frame(width: 22, height: 22)
                                .background(.orange, in: Circle())
                            Text(item)
                                .font(.subheadline)
                                .foregroundStyle(Color.white)
                        }
                    }
                }
            }

            if !guide.fiveMinRoutine.isEmpty {
                GuideSection(title: "5분 정리 루틴", icon: "timer", color: Color(red: 0.18, green: 0.95, blue: 0.75)) {
                    ForEach(Array(guide.fiveMinRoutine.enumerated()), id: \.offset) { i, step in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: "\(i + 1).circle.fill")
                                .foregroundStyle(Color(red: 0.18, green: 0.95, blue: 0.75))
                            Text(step)
                                .font(.subheadline)
                                .foregroundStyle(Color.white)
                        }
                    }
                }
            }

            if !guide.byCategory.isEmpty {
                GuideSection(title: "카테고리별 가이드", icon: "tag.fill", color: .purple) {
                    ForEach(Array(guide.byCategory.sorted(by: { $0.key < $1.key })), id: \.key) { cat, advice in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(cat)
                                .font(.caption.bold())
                                .foregroundStyle(.purple)
                            Text(advice)
                                .font(.caption)
                                .foregroundStyle(Color.white.opacity(0.55))
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
        .background(
            LinearGradient(
                colors: [color.opacity(0.1), color.opacity(0.04)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(color.opacity(0.25), lineWidth: 1)
        )
    }
}

// MARK: - Previews

#Preview("전체 결과") {
    NavigationStack {
        ResultView(image: PreviewData.sampleImage, response: PreviewData.sampleResponse)
    }
}

#Preview("가이드 없음") {
    let r = AnalyzeResponse(
        imageId: "preview_002",
        processedImageSize: [960, 720],
        summary: "간단한 분석 결과입니다.",
        objects: Array(PreviewData.sampleObjects.prefix(2)),
        guide: Guide(priorityTop3: [], fiveMinRoutine: [], byCategory: [:]),
        annotatedImageUrl: "",
        latencyMs: 500
    )
    return NavigationStack { ResultView(image: PreviewData.sampleImage, response: r) }
}

#Preview("DecisionBadge - 버릴 것") {
    DecisionBadge(decision: "discard").padding().background(Color(red: 0.04, green: 0.04, blue: 0.06))
}

#Preview("DecisionBadge - 제자리") {
    DecisionBadge(decision: "keep").padding().background(Color(red: 0.04, green: 0.04, blue: 0.06))
}

#Preview("DecisionBadge - 미정") {
    DecisionBadge(decision: nil).padding().background(Color(red: 0.04, green: 0.04, blue: 0.06))
}
