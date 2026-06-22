import SwiftUI
import PhotosUI

// MARK: - Dark theme palette
private extension Color {
    static let appBackground    = Color(red: 0.04, green: 0.04, blue: 0.06)   // ~#0A0A0F
    static let cardBackground   = Color(red: 0.11, green: 0.11, blue: 0.14)   // ~#1C1C23
    static let cardBorder       = Color.white.opacity(0.08)
    static let neonMint         = Color(red: 0.18, green: 0.95, blue: 0.75)   // #2EF3BE
    static let neonMintDim      = Color(red: 0.18, green: 0.95, blue: 0.75).opacity(0.18)
    static let labelPrimary     = Color.white
    static let labelSecondary   = Color.white.opacity(0.55)
    static let labelTertiary    = Color.white.opacity(0.30)
}

struct HomeView: View {

    // MARK: - Persistent settings
    @AppStorage("serverURL")   private var serverURL   = "http://100.120.175.117"
    @AppStorage("useDescribe") private var useDescribe = false

    // MARK: - State
    @State private var pickerItem: PhotosPickerItem?
    @State private var vm: HomeViewModel
    @State private var showCamera   = false
    @State private var showSettings = false

    private var isCameraAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    init(viewModel: HomeViewModel = HomeViewModel()) {
        _vm = State(initialValue: viewModel)
    }

    // MARK: - Body

    var body: some View {
        @Bindable var vm = vm
        ZStack {
            Color.appBackground.ignoresSafeArea()

            // Subtle radial glow behind header
            RadialGradient(
                colors: [Color.neonMint.opacity(0.12), Color.clear],
                center: .top,
                startRadius: 0,
                endRadius: 320
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 28) {
                    headerSection
                    imagePickerSection
                    analyzeButton
                    settingsCard
                }
                .padding(.top, 8)
                .padding(.bottom, 48)
            }
        }
        .preferredColorScheme(.dark)
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Color.appBackground, for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .principal) {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.caption.bold())
                        .foregroundStyle(Color.neonMint)
                    Text("정리 도우미")
                        .font(.headline.bold())
                        .foregroundStyle(Color.labelPrimary)
                }
            }
        }
        .sheet(item: $vm.analysisResult) { result in
            NavigationStack {
                ResultView(image: result.image, response: result.response)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("닫기") { vm.analysisResult = nil }
                                .foregroundStyle(Color.neonMint)
                        }
                    }
            }
        }
        .sheet(isPresented: $showCamera) {
            CameraCapture(capturedImage: $vm.selectedImage)
                .ignoresSafeArea()
        }
        .alert("오류", isPresented: Binding(
            get: { vm.errorMessage != nil },
            set: { if !$0 { vm.errorMessage = nil } }
        )) {
            Button("확인") { vm.errorMessage = nil }
        } message: {
            Text(vm.errorMessage ?? "")
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 14) {
            ZStack {
                // Outer glow ring
                Circle()
                    .fill(Color.neonMint.opacity(0.08))
                    .frame(width: 116, height: 116)
                // Inner ring
                Circle()
                    .fill(Color.neonMint.opacity(0.14))
                    .frame(width: 90, height: 90)
                // Icon
                Image(systemName: "sparkles.rectangle.stack.fill")
                    .font(.system(size: 38, weight: .medium))
                    .foregroundStyle(Color.neonMint)
            }
            .padding(.top, 12)
            .shadow(color: Color.neonMint.opacity(0.4), radius: 20, x: 0, y: 0)

            VStack(spacing: 5) {
                Text("공간 정리 AI")
                    .font(.title2.bold())
                    .foregroundStyle(Color.labelPrimary)
                Text("어질러진 공간을 사진으로 분석해 드려요")
                    .font(.subheadline)
                    .foregroundStyle(Color.labelSecondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Image Picker

    private var imagePickerSection: some View {
        VStack(spacing: 12) {
            PhotosPicker(selection: $pickerItem, matching: .images) {
                pickerLabel
            }
            .onChange(of: pickerItem) { _, item in
                Task { await vm.loadImage(from: item) }
            }

            if isCameraAvailable {
                Button {
                    showCamera = true
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "camera.fill")
                            .font(.subheadline)
                        Text("카메라로 직접 찍기")
                            .font(.subheadline.weight(.medium))
                    }
                    .foregroundStyle(Color.labelPrimary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .background(Color.cardBackground, in: RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .strokeBorder(Color.cardBorder, lineWidth: 1)
                    )
                }
            }
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private var pickerLabel: some View {
        if let img = vm.selectedImage {
            ZStack(alignment: .bottomTrailing) {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: .infinity)
                    .frame(height: 260)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .strokeBorder(Color.neonMint.opacity(0.35), lineWidth: 1)
                    )

                HStack(spacing: 5) {
                    Image(systemName: "photo.badge.plus")
                        .font(.caption.bold())
                    Text("변경")
                        .font(.caption.bold())
                }
                .foregroundStyle(Color.labelPrimary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(Color.black.opacity(0.55), in: Capsule())
                .overlay(Capsule().strokeBorder(Color.white.opacity(0.15), lineWidth: 0.5))
                .padding(14)
            }
            .shadow(color: Color.neonMint.opacity(0.2), radius: 20, x: 0, y: 6)
        } else {
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.cardBackground)
                .frame(maxWidth: .infinity)
                .frame(height: 220)
                .overlay {
                    VStack(spacing: 14) {
                        ZStack {
                            Circle()
                                .fill(Color.neonMint.opacity(0.1))
                                .frame(width: 70, height: 70)
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 30))
                                .foregroundStyle(Color.neonMint)
                        }
                        VStack(spacing: 3) {
                            Text("사진 선택하기")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Color.labelPrimary)
                            Text("갤러리에서 이미지를 가져옵니다")
                                .font(.caption)
                                .foregroundStyle(Color.labelSecondary)
                        }
                    }
                }
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.neonMint.opacity(0.4), Color.neonMint.opacity(0.1)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.5
                        )
                )
        }
    }

    // MARK: - Analyze Button

    private var analyzeButton: some View {
        Button {
            Task { await vm.analyze(serverURL: serverURL, useDescribe: useDescribe) }
        } label: {
            ZStack {
                if vm.isAnalyzing {
                    HStack(spacing: 10) {
                        ProgressView()
                            .tint(Color.appBackground)
                            .scaleEffect(0.9)
                        Text("AI 분석 중…")
                            .font(.headline)
                    }
                } else {
                    HStack(spacing: 8) {
                        Image(systemName: "wand.and.sparkles")
                            .font(.headline)
                        Text("AI로 분석 시작")
                            .font(.headline)
                    }
                }
            }
            .foregroundStyle(vm.canAnalyze ? Color.appBackground : Color.labelTertiary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background {
                if vm.canAnalyze {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.neonMint)
                } else {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.cardBackground)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .strokeBorder(Color.cardBorder, lineWidth: 1)
                        )
                }
            }
        }
        .disabled(!vm.canAnalyze)
        .padding(.horizontal)
        .shadow(
            color: vm.canAnalyze ? Color.neonMint.opacity(0.45) : .clear,
            radius: 18, x: 0, y: 6
        )
        .animation(.easeInOut(duration: 0.2), value: vm.canAnalyze)
    }

    // MARK: - Settings Card

    private var settingsCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.spring(duration: 0.3)) { showSettings.toggle() }
            } label: {
                HStack {
                    Label("서버 및 모델 설정", systemImage: "gearshape.fill")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Color.labelPrimary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption.bold())
                        .foregroundStyle(Color.labelSecondary)
                        .rotationEffect(.degrees(showSettings ? 90 : 0))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
            }

            if showSettings {
                Divider()
                    .background(Color.cardBorder)
                    .padding(.horizontal)

                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("서버 주소", systemImage: "server.rack")
                            .font(.caption.bold())
                            .foregroundStyle(Color.labelSecondary)

                        HStack(spacing: 8) {
                            TextField("https://xxxx-8000.app.github.dev", text: Binding(
                                get: { serverURL },
                                set: { serverURL = $0 }
                            ))
                            .textFieldStyle(.plain)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .keyboardType(.URL)
                            .font(.caption)
                            .foregroundStyle(Color.labelPrimary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.cardBorder, lineWidth: 1)
                            )

                            Button {
                                Task { await vm.checkHealth(serverURL: serverURL) }
                            } label: {
                                Image(systemName: "antenna.radiowaves.left.and.right")
                                    .font(.body)
                                    .frame(width: 38, height: 34)
                                    .background(vm.serverStatus.dotColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .strokeBorder(vm.serverStatus.dotColor.opacity(0.5), lineWidth: 1)
                                    )
                            }
                            .foregroundStyle(vm.serverStatus.dotColor)
                        }

                        HStack(spacing: 5) {
                            if case .checking = vm.serverStatus {
                                ProgressView().scaleEffect(0.65).tint(Color.neonMint)
                            } else {
                                Circle()
                                    .fill(vm.serverStatus.dotColor)
                                    .frame(width: 6, height: 6)
                            }
                            Text(vm.serverStatus.label)
                                .font(.caption2)
                                .foregroundStyle(Color.labelSecondary)
                        }
                    }

                    Divider().background(Color.cardBorder)

                    Toggle(isOn: Binding(
                        get: { useDescribe },
                        set: { useDescribe = $0 }
                    )) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Florence-2 Describe")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(Color.labelPrimary)
                            Text("속성 서술 포함 (느림) / 빠른 탐지만 (기본)")
                                .font(.caption)
                                .foregroundStyle(Color.labelSecondary)
                        }
                    }
                    .tint(Color.neonMint)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(Color.cardBackground, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(Color.cardBorder, lineWidth: 1)
        )
        .padding(.horizontal)
    }
}

// MARK: - Previews

#Preview("기본 상태") {
    NavigationStack { HomeView() }
}

#Preview("사진 선택됨") {
    let vm = HomeViewModel(serviceFactory: { _ in MockAPIService() })
    vm.selectedImage = PreviewData.sampleImage
    return NavigationStack { HomeView(viewModel: vm) }
}

#Preview("분석 중 (스피너)") {
    let mock = MockAPIService()
    mock.delay = .seconds(300)
    let vm = HomeViewModel(serviceFactory: { _ in mock })
    vm.selectedImage = PreviewData.sampleImage
    vm.isAnalyzing = true
    return NavigationStack { HomeView(viewModel: vm) }
}
