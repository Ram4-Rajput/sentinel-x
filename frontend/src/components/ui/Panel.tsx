import { ReactNode } from "react";
import { cn } from "@/lib/format";

export function Panel({
  title,
  right,
  children,
  className,
  bodyClassName,
}: {
  title?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("panel flex flex-col overflow-hidden", className)}>
      {title && (
        <header className="flex items-center justify-between px-3 py-2 border-b border-[#16203a]">
          <span className="panel-title">{title}</span>
          {right}
        </header>
      )}
      <div className={cn("flex-1 min-h-0 overflow-auto scroll-thin", bodyClassName)}>
        {children}
      </div>
    </section>
  );
}

export function Metric({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="panel-title">{label}</span>
      <span className="mono text-lg font-semibold" style={{ color }}>
        {value}
      </span>
      {sub && <span className="text-[10px] text-inkdim">{sub}</span>}
    </div>
  );
}
