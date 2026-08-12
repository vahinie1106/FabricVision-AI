import { redirect } from "next/navigation";

/** Original app had no Studio hub — AI Studio linked directly to Custom Garment. */
export default function StudioIndex() {
  redirect("/studio/custom-garment");
}
