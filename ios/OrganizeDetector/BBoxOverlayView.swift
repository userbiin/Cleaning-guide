import SwiftUI

/// Renders bbox rectangles + label pills on top of the displayed image.
/// Placed via Image(...).overlay(GeometryReader { geo in BBoxOverlayView(..., displaySize: geo.size) })
/// so geo.size always equals the actual rendered image dimensions.
struct BBoxOverlayView: View {
    let objects:         [DetectedObject]
    let displaySize:     CGSize   // actual on-screen image size from GeometryReader
    let processedSize:   CGSize   // backend's [w, h] (bbox coordinates are in this space)
    @Binding var selectedID: Int?

    // Scale factors from v2 §8.1:
    // rect.x = bbox[0] * sx,  rect.y = bbox[1] * sy
    private var sx: CGFloat { displaySize.width  / max(processedSize.width,  1) }
    private var sy: CGFloat { displaySize.height / max(processedSize.height, 1) }

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Canvas: boxes + label pills (non-interactive, sits on top of tap targets)
            Canvas { ctx, _ in
                for obj in objects where obj.bbox.count == 4 {
                    let rect      = scaledRect(obj)
                    let color     = obj.decisionColor
                    let isSelected = selectedID == obj.id

                    // Bounding box
                    ctx.stroke(
                        Path(roundedRect: rect, cornerRadius: 4),
                        with: .color(color),
                        lineWidth: isSelected ? 3.5 : 2
                    )

                    // Label pill
                    let labelText = "\(obj.label) \(Int(obj.confidence * 100))%"
                    let pillW = CGFloat(labelText.count) * 6.4 + 10
                    let pillH: CGFloat = 18
                    let pillY = max(rect.minY - pillH - 2, 0)
                    let pillRect = CGRect(x: rect.minX, y: pillY, width: pillW, height: pillH)

                    ctx.fill(
                        Path(RoundedRect(rect: pillRect, cornerSize: CGSize(width: 4, height: 4))),
                        with: .color(color)
                    )
                    ctx.draw(
                        ctx.resolve(Text(labelText).font(.system(size: 10, weight: .semibold)).foregroundStyle(.white)),
                        at: CGPoint(x: pillRect.midX, y: pillRect.midY),
                        anchor: .center
                    )
                }
            }
            .allowsHitTesting(false)

            // Transparent tap targets covering each bbox
            ForEach(objects.filter { $0.bbox.count == 4 }) { obj in
                let rect = scaledRect(obj)
                Color.clear
                    .frame(width: rect.width, height: rect.height)
                    .offset(x: rect.minX, y: rect.minY)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        withAnimation(.spring(duration: 0.25)) {
                            selectedID = selectedID == obj.id ? nil : obj.id
                        }
                    }
            }
        }
    }

    private func scaledRect(_ obj: DetectedObject) -> CGRect {
        CGRect(
            x:      CGFloat(obj.bbox[0]) * sx,
            y:      CGFloat(obj.bbox[1]) * sy,
            width:  CGFloat(obj.bbox[2] - obj.bbox[0]) * sx,
            height: CGFloat(obj.bbox[3] - obj.bbox[1]) * sy
        )
    }
}
