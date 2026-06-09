import SwiftUI
import PhotosUI

struct HomeView: View {

    // MARK: - Persistent settings (View에 유지 — @AppStorage는 SwiftUI 전용 래퍼)
    @AppStorage("serverURL")   private var serverURL   = "http://100.x.x.x:8000"
    @AppStorage("useDescribe") private var useDescribe = false

    // MARK: - State
    @State private var pickerItem: PhotosPickerItem?
    @State private var vm: HomeViewModel
    @State private var showCamera = false

    private var isCameraAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    // MARK: - Init (ViewModel 주입 지원 → Preview/테스트에서 Mock 사용 가능)
    init(viewModel: HomeViewModel = HomeViewModel()) {
        _vm = State(initialValue: viewModel)
    }

    // MARK: - Body

    var body: some View {
        @Bindable var vm = vm
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
        .sheet(item: $vm.analysisResult) { result in
            NavigationStack {
                ResultView(image: result.image, response: result.response)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("닫기") { vm.analysisResult = nil }
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
        VStack(spacing: 8) {
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
                    Label("카메라로 직접 찍기", systemImage: "camera.fill")
                        .font(.subheadline.weight(.medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.bordered)
                .tint(.blue)
            }
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private var pickerLabel: some View {
        if let img = vm.selectedImage {
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

            HStack(spacing: 8) {
                TextField("https://xxxx-8000.app.github.dev", text: $serverURL)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)

                Button {
                    Task { await vm.checkHealth(serverURL: serverURL) }
                } label: {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .font(.body)
                }
                .buttonStyle(.bordered)
                .tint(vm.serverStatus.dotColor)
            }

            // 연결 상태 표시
            HStack(spacing: 5) {
                if case .checking = vm.serverStatus {
                    ProgressView().scaleEffect(0.7)
                } else {
                    Circle()
                        .fill(vm.serverStatus.dotColor)
                        .frame(width: 7, height: 7)
                }
                Text(vm.serverStatus.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

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
        Button {
            Task { await vm.analyze(serverURL: serverURL, useDescribe: useDescribe) }
        } label: {
            Group {
                if vm.isAnalyzing {
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
        .disabled(!vm.canAnalyze)
        .padding(.horizontal)
    }
}

// MARK: - Previews

#Preview("기본 상태") {
    NavigationStack {
        HomeView()
    }
}

#Preview("사진 선택됨") {
    let vm = HomeViewModel(serviceFactory: { _ in MockAPIService() })
    vm.selectedImage = PreviewData.sampleImage
    return NavigationStack {
        HomeView(viewModel: vm)
    }
}

#Preview("분석 중 (스피너)") {
    let mock = MockAPIService()
    mock.delay = .seconds(300) // 완료되지 않아 스피너 유지
    let vm = HomeViewModel(serviceFactory: { _ in mock })
    vm.selectedImage = PreviewData.sampleImage
    vm.isAnalyzing = true
    return NavigationStack {
        HomeView(viewModel: vm)
    }
}
