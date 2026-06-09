// OrganizeDetectorApp.swift
//
// ── Xcode 신규 프로젝트 설정 ────────────────────────────────────────────────────
//  1. New Project → iOS → App, Product Name: OrganizeDetector, Interface: SwiftUI
//  2. Minimum Deployments: iOS 17.0
//  3. Info.plist / Signing & Capabilities → Info 탭에 아래 항목 추가:
//       NSPhotoLibraryUsageDescription  →  "사진에서 정리할 공간을 선택합니다."
//       NSCameraUsageDescription        →  "카메라로 정리할 공간을 촬영합니다."
//       App Transport Security Settings → Allow Arbitrary Loads → YES
//       (데모 전용. 배포 시 NSExceptionDomains에 Codespace/Tailscale 도메인만 허용할 것)
//  4. Codespace 백엔드 연결:
//       uvicorn app.main:app --host 0.0.0.0 --port 8000  (Codespace 터미널)
//       VS Code PORTS 탭 → 8000번 → Port Visibility: Public
//       앱 서버 주소 필드에 https://xxxx-8000.app.github.dev 입력
// ────────────────────────────────────────────────────────────────────────────────

import SwiftUI

@main
struct OrganizeDetectorApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                HomeView()
            }
        }
    }
}
