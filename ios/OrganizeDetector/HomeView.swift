import SwiftUI
import PhotosUI

// Bundles UIImage + AnalyzeResponse so .sheet(item:) can present both together.
private struct AnalysisPacket: Identifiable {
    let id = UUID()
    let image: UIImage
    let response: AnalyzeResponse
}

struct HomeView: View {
    // Persisted across launches via UserDefaults
    @AppStorage("serverURL")   private var serverURL   = "http://100.x.x.x:8000"
    @AppStorage("useDescribe") private var useDescribe = false

    @State private var pickerItem:  PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var isAnalyzing  = false
    @State private var packet:       AnalysisPacket?    // non-nil → sheet opens
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                headerSection
                imagePickerSection
                settingsCard
                analyzeButton
            }
            .padding(.bottom, 40)
        }
        .navigationTitle("정리 도우미")
        .navigationBarTitleDisplayMode(.large)
        .sheet(item: $packet) { pkt in
            NavigationStack {
                ResultView(image: pkt.image, response: pkt.response)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("닫기") { packet = nil }
                        }
                    }
            }
        }
        .alert("오류", isPresented: .constant(errorMessage != nil)) {
            Button("확인") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    // MARK: - Sub-views

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles.rectangle.stack.fill")
                .font(.system(size: 60))
                .foregroundStyle(.tint)
                .padding(.top, 8)
            Text("어질러진 공간을 사진으로 찍어보세요")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }

    private var imagePickerSection: some View {
        PhotosPicker(selection: $pickerItem, matching: .images) {
            pickerLabel
        }
        .padding(.horizontal)
        .onChange(of: pickerItem) { _, item in
            Task {
                guard let data = try? await item?.loadTransferable(type: Data.self),
                      let img  = UIImage(data: data) else { return }
                await MainActor.run { selectedImage = img }
            }
        }
    }

    @ViewBuilder
    private var pickerLabel: some View {
        if let img = selectedImage {
            Image(uiImage: img)
                .resizable()
                .scaledToFill()
                .frame(maxWidth: .infinity)
                .frame(height: 240)
                .clipped()
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "pencil.circle.fill")
                        .symbolRenderingMode(.multicolor)
                        .font(.title)
                        .padding(12)
                }
        } else {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.secondarySystemBackground))
                .frame(maxWidth: .infinity)
                .frame(height: 200)
                .overlay {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary)
                        Text("사진 선택하기")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
        }
    }

    private var settingsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("서버 주소", systemImage: "server.rack")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            TextField("http://100.x.x.x:8000", text: $serverURL)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)

            Divider()

            Toggle(isOn: $useDescribe) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Florence-2 Describe")
                        .font(.subheadline.weight(.medium))
                    Text("ON: 속성 서술 추가 (느림) / OFF: 빠른 탐지만 (대조군)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal)
    }

    private var analyzeButton: some View {
        Button(action: analyze) {
            Group {
                if isAnalyzing {
                    HStack(spacing: 10) {
                        ProgressView().tint(.white)
                        Text("분석 중…")
                    }
                } else {
                    Label("분석 시작", systemImage: "magnifyingglass")
                }
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .disabled(selectedImage == nil || isAnalyzing)
        .padding(.horizontal)
    }

    // MARK: - Action

    private func analyze() {
        guard let image = selectedImage else { return }
        isAnalyzing = true

        let service  = APIService(baseURL: serverURL)
        let describe = useDescribe

        Task {
            do {
                let response = try await service.analyze(image: image, describe: describe)
                await MainActor.run {
                    packet     = AnalysisPacket(image: image, response: response)
                    isAnalyzing = false
                }
            } catch let urlErr as URLError {
                await MainActor.run {
                    errorMessage = "네트워크 오류 (\(urlErr.code.rawValue)). 서버가 실행 중인지 확인하세요."
                    isAnalyzing  = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isAnalyzing  = false
                }
            }
        }
    }
}
