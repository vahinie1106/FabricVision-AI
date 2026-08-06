import { Card } from "@/components/ui/Card";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import Link from "next/link";

export default function About() {
  return (
    <div className="max-w-4xl mx-auto w-full pb-20">
      <div className="mb-12">
        <h1 className="text-4xl md:text-5xl font-bold mb-6">About FabricVision-AI</h1>
        <p className="text-lg text-[#767676] leading-relaxed">
          FabricVision-AI is an advanced fashion intelligence platform developed for <strong>Fabriplay</strong>. 
          Our mission is to democratize high-end fashion design by seamlessly blending state-of-the-art generative AI with professional workflows.
        </p>
      </div>

      <div className="space-y-8">
        <Card className="p-8">
          <h2 className="text-2xl font-bold mb-4">The Platform</h2>
          <p className="text-[#767676] mb-6 leading-relaxed">
            By shifting from a monolithic backend structure to a loosely coupled, API-driven architecture, FabricVision-AI empowers designers to generate, visualize, and analyze fashion at scale. 
            The platform is broken down into three independent modules designed to handle the core lifecycle of a fashion product.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start">
              <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
              <p className="text-sm font-medium">Custom Garment Synthesis</p>
            </div>
            <div className="flex items-start">
              <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
              <p className="text-sm font-medium">High-Fidelity Virtual Try-On</p>
            </div>
            <div className="flex items-start">
              <CheckCircle2 className="w-5 h-5 text-[#D8B4E2] mr-3 mt-0.5 flex-shrink-0" />
              <p className="text-sm font-medium">Semantic Analysis (Coming Soon)</p>
            </div>
          </div>
        </Card>

        <Card className="p-8">
          <h2 className="text-2xl font-bold mb-4">Core AI Models</h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-bold mb-2">FLUX Kontext</h3>
              <p className="text-sm text-[#767676] leading-relaxed">
                Utilized for generating hyper-realistic garments from raw fabric textures. The model handles complex lighting, draping, and material translation with unparalleled accuracy.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-2">CatVTON</h3>
              <p className="text-sm text-[#767676] leading-relaxed">
                Our chosen engine for Virtual Try-On. CatVTON maps existing garments onto target personas while preserving the original clothing structure and fabric integrity.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-2">Qwen2.5-VL</h3>
              <p className="text-sm text-[#767676] leading-relaxed">
                A massive vision-language model that powers our upcoming Semantic Analysis engine, capable of understanding complex fashion taxonomies and extracting rich metadata.
              </p>
            </div>
          </div>
        </Card>

        <div className="flex justify-center mt-12">
          <Link href="/studio/custom-garment" className="inline-flex items-center justify-center px-8 py-4 bg-[#1A1A1A] text-white rounded-xl font-semibold hover:bg-[#333333] transition-transform hover:scale-[1.02] shadow-[0_4px_12px_rgba(26,26,26,0.2)]">
            Explore the Studio <ArrowRight className="w-5 h-5 ml-2" />
          </Link>
        </div>
      </div>
    </div>
  );
}
