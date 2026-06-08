// OrganizeDetectorApp.swift
//
// ── Xcode 신규 프로젝트 설정 ────────────────────────────────────────────────────
//  1. New Project → iOS → App, Product Name: OrganizeDetector, Interface: SwiftUI
//  2. Minimum Deployments: iOS 17.0
//  3. Signing & Capabilities → Info 탭에 아래 두 항목 추가:
//       NSPhotoLibraryUsageDescription  →  "사진에서 정리할 공간을 선택합니다."
//       App Transport Security Settings → Allow Arbitrary Loads → YES
//       (데모 전용. 배포 시 NSExceptionDomains에 Tailscale IP만 허용할 것)
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
