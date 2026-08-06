"""
Pydantic Schemas for Garment Metadata Management.

Defines strongly-typed models covering:
- Garment Identity (category, gender, season, occasion)
- Physical Attributes (fabric, texture, color, pattern)
- Construction Attributes (neckline, sleeve, silhouette, fit)
- Style Attributes (aesthetic, trend, fashion_category / fashion_style)
- GarmentMetadata (canonical container with auto-timestamp & ID validation)
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class GarmentIdentity(BaseModel):
    category: str = Field(..., description="Master garment category e.g. upper_wear, lower_wear, dresses")
    gender: str = Field(..., description="Target gender e.g. men, women, unisex")
    season: str = Field(..., description="Target season e.g. summer, winter, all_season")
    occasion: str = Field(..., description="Primary wearing occasion e.g. casual, formal, party")


class PhysicalAttributes(BaseModel):
    fabric: str = Field(..., description="Primary fabric/material type e.g. cotton, silk, denim")
    texture: str = Field(..., description="Surface texture e.g. smooth, ribbed, rough")
    color: List[str] = Field(..., description="List of visual colors e.g. ['white', 'blue']")
    pattern: str = Field(..., description="Visual pattern e.g. solid, striped, floral")

    @field_validator("color", mode="before")
    def ensure_color_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return [str(item) for item in v]
        return ["unknown"]


class Construction(BaseModel):
    neckline: str = Field(..., description="Neckline style e.g. crew, v_neck, collared")
    sleeve: str = Field(..., description="Sleeve length e.g. short, long, sleeveless")
    silhouette: str = Field(..., description="Overall garment silhouette e.g. regular, fitted, a_line")
    fit: str = Field(..., description="Garment fit type e.g. slim, regular, loose")


class Style(BaseModel):
    aesthetic: str = Field(..., description="Fashion aesthetic e.g. minimalist, streetwear, vintage")
    trend: str = Field(..., description="Fashion trend indicator e.g. classic, modern, retro")
    fashion_category: str = Field(..., description="High-level style category or sub-category e.g. basics, jeans")


class GarmentMetadata(BaseModel):
    garment_id: str = Field(..., description="Unique identifier for the garment, e.g. garment_000001")
    identity: GarmentIdentity
    physical: PhysicalAttributes
    construction: Construction
    style: Style

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_dataset: Optional[str] = Field(default="FabricVision-AI", description="Origin dataset adapter name")

    @field_validator("garment_id")
    def validate_id_format(cls, v: str) -> str:
        if not v.startswith("garment_"):
            raise ValueError("Garment ID must start with prefix 'garment_'")
        return v

    @classmethod
    def from_legacy_dict(cls, raw_data: Dict[str, Any], garment_id: str = "garment_000001") -> "GarmentMetadata":
        """
        Adapter method to map legacy Qwen semantic analysis dictionary output
        into canonical GarmentMetadata Pydantic model.
        """
        gi = raw_data.get("garment_identity", {})
        cls_info = raw_data.get("classification", {})
        pa = raw_data.get("physical_attributes", {})
        va = raw_data.get("visual_attributes", {})
        sf = raw_data.get("shape_and_fit", {})
        st = raw_data.get("style", {})

        # Handle nested or direct schemas
        if "identity" in raw_data and "physical" in raw_data:
            return cls(**raw_data)

        colors = va.get("colors", pa.get("colors", ["black"]))
        if isinstance(colors, str):
            colors = [colors]

        return cls(
            garment_id=raw_data.get("garment_id", garment_id),
            identity=GarmentIdentity(
                category=cls_info.get("category", gi.get("category", "upper_wear")),
                gender=gi.get("gender", "unisex"),
                season=st.get("season", "all_season"),
                occasion=st.get("occasion", "casual"),
            ),
            physical=PhysicalAttributes(
                fabric=pa.get("material", pa.get("fabric", "cotton")),
                texture=pa.get("fabric_textures", pa.get("texture", "smooth")),
                color=colors if colors else ["black"],
                pattern=va.get("patterns", pa.get("pattern", "solid")),
            ),
            construction=Construction(
                neckline=sf.get("neckline", "crew"),
                sleeve=sf.get("sleeves", sf.get("sleeve", "short")),
                silhouette=sf.get("silhouette", "regular"),
                fit=sf.get("fit", "regular"),
            ),
            style=Style(
                aesthetic=st.get("aesthetic", "casual"),
                trend=st.get("trend", "classic"),
                fashion_category=cls_info.get("subcategory", st.get("fashion_category", "basics")),
            ),
            source_dataset=raw_data.get("source_dataset", "Qwen2.5-VL"),
        )
