"use client";

import * as React from "react";
import * as AspectRatioPrimitive from "@radix-ui/react-aspect-ratio";

type AspectRatioProps = React.ComponentProps<typeof AspectRatioPrimitive.Root>;

function AspectRatio({ children, ...props }: AspectRatioProps) {
  return (
    <AspectRatioPrimitive.Root data-slot="aspect-ratio" {...props}>
      {children}
    </AspectRatioPrimitive.Root>
  );
}

export { AspectRatio };
