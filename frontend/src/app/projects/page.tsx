"use client";

import { EmptyState } from "@/components/feedback/EmptyState";
import { FolderOpen } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";

// Dummy data for future state
// const dummyProjects = [];

export default function ProjectsDashboard() {
  const hasProjects = false;

  return (
    <div className="max-w-[1200px] mx-auto w-full">
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-4xl font-bold text-[#1A1A1A]">My Projects</h1>
          <p className="text-[#767676] mt-2">Manage your AI fashion creations and metadata.</p>
        </div>
        <Link href="/studio/custom-garment">
          <Button>New Creation</Button>
        </Link>
      </div>

      {!hasProjects ? (
        <EmptyState
          icon={FolderOpen}
          title="No creations yet"
          description="Your workspace is empty. Start designing new garments or trying them on virtually to build out your fashion catalog."
          actionLabel="Create your first AI garment"
          onAction={() => window.location.href = "/studio/custom-garment"}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Future populated state would map over project cards here */}
        </div>
      )}
    </div>
  );
}
