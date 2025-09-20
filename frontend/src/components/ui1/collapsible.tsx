"use client";

import * as React from "react";
import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";

type CollapsibleProps = React.ComponentProps<typeof CollapsiblePrimitive.Root>;
type CollapsibleTriggerProps = React.ComponentProps<
  typeof CollapsiblePrimitive.Trigger
>;
type CollapsibleContentProps = React.ComponentProps<
  typeof CollapsiblePrimitive.Content
>;

function Collapsible({ children, ...props }: CollapsibleProps) {
  return (
    <CollapsiblePrimitive.Root data-slot="collapsible" {...props}>
      {children}
    </CollapsiblePrimitive.Root>
  );
}

function CollapsibleTrigger({ children, ...props }: CollapsibleTriggerProps) {
  return (
    <CollapsiblePrimitive.Trigger data-slot="collapsible-trigger" {...props}>
      {children}
    </CollapsiblePrimitive.Trigger>
  );
}

function CollapsibleContent({ children, ...props }: CollapsibleContentProps) {
  return (
    <CollapsiblePrimitive.Content data-slot="collapsible-content" {...props}>
      {children}
    </CollapsiblePrimitive.Content>
  );
}

export { Collapsible, CollapsibleTrigger, CollapsibleContent };
