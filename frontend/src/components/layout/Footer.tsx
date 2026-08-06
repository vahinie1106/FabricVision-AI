import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full border-t border-gray-200 bg-white mt-16">
      <div className="max-w-[1440px] mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1">
            <span className="font-[family-name:var(--font-outfit)] text-xl font-semibold tracking-tight text-[#1A1A1A]">
              FabricVision<span className="text-[#D8B4E2]">-AI</span>
            </span>
            <p className="mt-4 text-sm text-[#767676] leading-relaxed">
              Democratizing high-fashion creation through artificial intelligence. Powered by Fabriplay.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-[#1A1A1A] mb-4 text-sm uppercase tracking-wider">Product</h4>
            <ul className="space-y-3 text-sm text-[#767676]">
              <li><Link href="/studio/custom-garment" className="hover:text-[#1A1A1A]">Custom Garment Generator</Link></li>
              <li><Link href="/studio/virtual-tryon" className="hover:text-[#1A1A1A]">Virtual Try-On</Link></li>
              <li><Link href="/studio/semantic-analysis" className="hover:text-[#1A1A1A]">Semantic Analysis (Beta)</Link></li>
              <li><Link href="/pricing" className="hover:text-[#1A1A1A]">Pricing</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-[#1A1A1A] mb-4 text-sm uppercase tracking-wider">Resources</h4>
            <ul className="space-y-3 text-sm text-[#767676]">
              <li><Link href="/docs" className="hover:text-[#1A1A1A]">Documentation</Link></li>
              <li><Link href="/api" className="hover:text-[#1A1A1A]">API Reference</Link></li>
              <li><Link href="/privacy" className="hover:text-[#1A1A1A]">Privacy Policy</Link></li>
              <li><Link href="/terms" className="hover:text-[#1A1A1A]">Terms of Service</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-[#1A1A1A] mb-4 text-sm uppercase tracking-wider">Connect</h4>
            <ul className="space-y-3 text-sm text-[#767676]">
              <li><Link href="#" className="hover:text-[#1A1A1A]">Twitter</Link></li>
              <li><Link href="#" className="hover:text-[#1A1A1A]">LinkedIn</Link></li>
              <li><Link href="#" className="hover:text-[#1A1A1A]">Instagram</Link></li>
              <li><Link href="#" className="hover:text-[#1A1A1A]">Contact Support</Link></li>
            </ul>
          </div>
        </div>
        <div className="mt-12 pt-8 border-t border-gray-100 flex flex-col md:flex-row justify-between items-center text-xs text-[#767676]">
          <p>© {new Date().getFullYear()} Fabriplay. All rights reserved.</p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <span>Powered by FLUX & CatVTON</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
