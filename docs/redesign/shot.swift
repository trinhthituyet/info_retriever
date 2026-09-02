import AppKit
import WebKit

// Usage: swift shot.swift <input.html> <output.png> <width> <height> [scale]
let args = CommandLine.arguments
let inPath = args[1], outPath = args[2]
let w = Double(args[3])!, h = Double(args[4])!
let scale = args.count > 5 ? Double(args[5])! : 2.0

let app = NSApplication.shared
app.setActivationPolicy(.accessory)

final class Shooter: NSObject, WKNavigationDelegate {
    let web: WKWebView
    let win: NSWindow
    override init() {
        let cfg = WKWebViewConfiguration()
        let rect = NSRect(x: 0, y: 0, width: w, height: h)
        web = WKWebView(frame: rect, configuration: cfg)
        win = NSWindow(contentRect: rect, styleMask: [.borderless],
                       backing: .buffered, defer: false)
        win.contentView = web
        win.setIsVisible(false)
        super.init()
        web.navigationDelegate = self
    }
    func load() {
        let url = URL(fileURLWithPath: inPath)
        web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    }
    func webView(_ wv: WKWebView, didFinish nav: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.snap() }
    }
    func snap() {
        let cfg = WKSnapshotConfiguration()
        cfg.rect = CGRect(x: 0, y: 0, width: w, height: h)
        cfg.snapshotWidth = NSNumber(value: w * scale)
        web.takeSnapshot(with: cfg) { image, err in
            guard let image, let tiff = image.tiffRepresentation,
                  let rep = NSBitmapImageRep(data: tiff),
                  let png = rep.representation(using: .png, properties: [:]) else {
                FileHandle.standardError.write("snapshot failed: \(String(describing: err))\n".data(using: .utf8)!)
                exit(1)
            }
            try! png.write(to: URL(fileURLWithPath: outPath))
            print("wrote \(outPath) \(rep.pixelsWide)x\(rep.pixelsHigh)")
            exit(0)
        }
    }
}

let s = Shooter()
s.load()
app.run()
