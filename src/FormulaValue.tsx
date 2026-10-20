import React, {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { type Calculation, numberText } from "./calculations";

type Props = {
  calculation?: Calculation;
  children?: React.ReactNode;
  as?: string;
  formulaKey?: string;
  [key: string]: any;
};
/** Hover, keyboard focus and touch share the same viewport-bounded explanation. */
export default function FormulaValue({
  calculation,
  children,
  as = "span",
  formulaKey,
  ...props
}: Props) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0, ready: false });
  const trigger = useRef<Element>(null),
    popup = useRef<HTMLDivElement>(null),
    closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const id = useId();
  const close = () => {
    clearTimeout(closeTimer.current);
    setOpen(false);
  };
  const show = () => {
    clearTimeout(closeTimer.current);
    if (!open) {
      setPosition((p) => ({ ...p, ready: false }));
      setOpen(true);
    }
  };
  const delayClose = () => {
    closeTimer.current = setTimeout(close, 160);
  };
  useEffect(() => () => clearTimeout(closeTimer.current), []);
  useLayoutEffect(() => {
    if (!open || !trigger.current || !popup.current) return;
    const box = trigger.current.getBoundingClientRect(),
      tip = popup.current.getBoundingClientRect();
    const left = Math.max(
      12,
      Math.min(box.left, window.innerWidth - tip.width - 12),
    );
    const below = box.bottom + 9;
    const top =
      below + tip.height <= window.innerHeight - 12
        ? below
        : Math.max(12, box.top - tip.height - 9);
    setPosition({ left, top, ready: true });
  }, [open, calculation]);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    const pointer = (e: PointerEvent) => {
      if (
        !trigger.current?.contains(e.target as Node) &&
        !popup.current?.contains(e.target as Node)
      )
        close();
    };
    const scroll = (e: Event) => {
      if (!popup.current?.contains(e.target as Node)) close();
    };
    document.addEventListener("keydown", key);
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("keydown", key);
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);
  const Tag = as as React.ElementType;
  if (!calculation) return <Tag {...props}>{children}</Tag>;
  return (
    <>
      <Tag
        {...props}
        ref={trigger}
        className={[props.className, "formula-value"].filter(Boolean).join(" ")}
        tabIndex={0}
        role="button"
        aria-label={`Formula for ${calculation.title}`}
        aria-describedby={open ? id : undefined}
        data-formula-key={formulaKey}
        onMouseEnter={show}
        onMouseLeave={delayClose}
        onFocus={show}
        onBlur={delayClose}
        onClick={(e: React.MouseEvent) => {
          e.stopPropagation();
          if (!open) show();
        }}
        onKeyDown={(e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            show();
          }
          if (e.key === "Escape") close();
        }}
      >
        {children}
      </Tag>
      {open &&
        createPortal(
          <div
            id={id}
            ref={popup}
            role="tooltip"
            className="formula-tooltip"
            style={{
              top: position.top,
              left: position.left,
              visibility: position.ready ? "visible" : "hidden",
            }}
            onMouseEnter={() => clearTimeout(closeTimer.current)}
            onMouseLeave={delayClose}
          >
            <div className="formula-tooltip-heading">
              {calculation.title}
              <small>Calculation from this run</small>
            </div>
            <pre>{calculation.formula}</pre>
            <div className="formula-tooltip-details">
              <dl>
                {calculation.variables.map((row, i) => (
                  <div key={row.symbol + i}>
                    <dt>
                      <code>{row.symbol}</code>
                      <span>{row.label}</span>
                    </dt>
                    <dd>
                      {numberText(row.value)}
                      {row.unit && <small>{row.unit}</small>}
                    </dd>
                  </div>
                ))}
              </dl>
              {calculation.note && <p>{calculation.note}</p>}
            </div>
            <div className="formula-result">
              <span>Result</span>
              <strong>
                {numberText(calculation.result)}{" "}
                <small>{calculation.unit}</small>
              </strong>
            </div>
            <div className="formula-tooltip-hint">
              Esc or tap outside to close
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
