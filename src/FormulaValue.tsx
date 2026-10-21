import React, {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { type Calculation, numberText } from "./calculations";

export const RUN_CHANGED_EVENT = "servescope:run-changed";

type Props = {
  calculation?: Calculation;
  children?: React.ReactNode;
  as?: string;
  formulaKey?: string;
  [key: string]: any;
};
type Step = { calculation: Calculation; anchor: DOMRect; key: string };

/** One portal owns the entire explanation path: entering a child keeps its ancestors open. */
export default function FormulaValue({
  calculation,
  children,
  as = "span",
  formulaKey,
  ...props
}: Props) {
  const [steps, setSteps] = useState<Step[]>([]);
  const trigger = useRef<Element>(null),
    cascade = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const openTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const id = useId();
  const cancelClose = () => clearTimeout(closeTimer.current);
  const close = () => {
    clearTimeout(openTimer.current);
    cancelClose();
    setSteps([]);
  };
  const delayClose = () => {
    clearTimeout(openTimer.current);
    closeTimer.current = setTimeout(close, 350);
  };
  const show = () => {
    cancelClose();
    if (!steps.length && calculation && trigger.current)
      setSteps([
        {
          calculation,
          anchor: trigger.current.getBoundingClientRect(),
          key: "root",
        },
      ]);
  };
  const branch = (
    depth: number,
    key: string,
    make: () => Calculation,
    target: HTMLElement,
    delayed: boolean,
  ) => {
    cancelClose();
    clearTimeout(openTimer.current);
    const anchor = target.getBoundingClientRect();
    const open = () =>
      setSteps((previous) =>
        previous[depth + 1]?.key === key
          ? previous
          : [
              ...previous.slice(0, depth + 1),
              { calculation: make(), anchor, key },
            ],
      );
    if (delayed) openTimer.current = setTimeout(open, 140);
    else open();
  };
  useEffect(
    () => () => {
      cancelClose();
      clearTimeout(openTimer.current);
    },
    [],
  );
  useEffect(() => {
    if (!steps.length) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        clearTimeout(openTimer.current);
        cancelClose();
        setSteps((previous) => previous.slice(0, -1));
      }
    };
    const pointer = (e: PointerEvent) => {
      if (
        !trigger.current?.contains(e.target as Node) &&
        !cascade.current?.contains(e.target as Node)
      )
        close();
    };
    const scroll = (e: Event) => {
      if (!cascade.current?.contains(e.target as Node)) close();
    };
    document.addEventListener("keydown", key);
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", close);
    window.addEventListener(RUN_CHANGED_EVENT, close);
    return () => {
      document.removeEventListener("keydown", key);
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", close);
      window.removeEventListener(RUN_CHANGED_EVENT, close);
    };
  }, [steps.length]);
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
        aria-describedby={steps.length ? id : undefined}
        data-formula-key={formulaKey}
        onMouseEnter={show}
        onMouseLeave={delayClose}
        onFocus={show}
        onBlur={(e: React.FocusEvent) => {
          if (!cascade.current?.contains(e.relatedTarget as Node)) delayClose();
        }}
        onClick={(e: React.MouseEvent) => {
          e.stopPropagation();
          show();
        }}
        onKeyDown={(e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            show();
          }
        }}
      >
        {children}
      </Tag>
      {!!steps.length &&
        createPortal(
          <div
            ref={cascade}
            className="formula-cascade"
            onMouseEnter={cancelClose}
            onMouseLeave={delayClose}
            onFocusCapture={cancelClose}
            onBlurCapture={(e) => {
              if (
                !cascade.current?.contains(e.relatedTarget as Node) &&
                !trigger.current?.contains(e.relatedTarget as Node)
              )
                delayClose();
            }}
          >
            {steps.map((step, depth) => (
              <FormulaCard
                key={depth + ":" + step.key}
                step={step}
                depth={depth}
                steps={steps}
                id={depth ? id + "-" + depth : id}
                onBranch={branch}
                onBack={() => setSteps((s) => s.slice(0, depth))}
                onClose={close}
                onCrumb={(index) => setSteps((s) => s.slice(0, index + 1))}
                onCancelOpen={() => clearTimeout(openTimer.current)}
              />
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}

function FormulaCard({
  step,
  depth,
  steps,
  id,
  onBranch,
  onBack,
  onClose,
  onCrumb,
  onCancelOpen,
}: {
  step: Step;
  depth: number;
  steps: Step[];
  id: string;
  onBranch: (
    depth: number,
    key: string,
    make: () => Calculation,
    target: HTMLElement,
    delayed: boolean,
  ) => void;
  onBack: () => void;
  onClose: () => void;
  onCrumb: (index: number) => void;
  onCancelOpen: () => void;
}) {
  const popup = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 12, left: 12, ready: false });
  const narrow = window.innerWidth < 900;
  const c = step.calculation;
  const hidden = depth < steps.length - (narrow ? 1 : 2);
  useLayoutEffect(() => {
    if (!popup.current) return;
    const tip = popup.current.getBoundingClientRect(),
      a = step.anchor;
    let left = Math.max(
      12,
      Math.min(a.left, window.innerWidth - tip.width - 12),
    );
    let top =
      a.bottom + 9 + tip.height <= window.innerHeight - 12
        ? a.bottom + 9
        : Math.max(12, a.top - tip.height - 9);
    // Prefer a side position when a tall card cannot fit above or below the
    // trigger. Covering the trigger would accidentally hover a term on open.
    if (
      !depth &&
      !narrow &&
      a.bottom + 9 + tip.height > window.innerHeight - 12 &&
      a.top - tip.height - 9 < 12
    ) {
      if (a.right + 10 + tip.width <= window.innerWidth - 12)
        left = a.right + 10;
      else if (a.left - tip.width - 10 >= 12) left = a.left - tip.width - 10;
      top = Math.max(12, Math.min(a.top, window.innerHeight - tip.height - 12));
    }
    if (depth) {
      const parent = popup.current.parentElement
        ?.querySelector(`[data-cascade-depth="${depth - 1}"]`)
        ?.getBoundingClientRect();
      if (parent && !narrow) {
        left =
          parent.right + 10 + tip.width <= window.innerWidth - 12
            ? parent.right + 10
            : Math.max(12, parent.left - tip.width - 10);
        top = Math.max(
          12,
          Math.min(a.top, window.innerHeight - tip.height - 12),
        );
      }
    }
    if (narrow) {
      left = (window.innerWidth - tip.width) / 2;
      top = Math.max(12, Math.min(70, window.innerHeight - tip.height - 12));
    }
    setPosition({ left, top, ready: true });
  }, [step, depth]);
  return (
    <div
      ref={popup}
      role="tooltip"
      id={id}
      data-cascade-depth={depth}
      data-calculation-title={c.title}
      className="formula-tooltip"
      aria-hidden={hidden || undefined}
      style={{
        top: position.top,
        left: position.left,
        visibility: position.ready ? "visible" : "hidden",
        zIndex: 10000 + depth,
        display: hidden ? "none" : undefined,
      }}
    >
      <nav className="formula-path" aria-label="Explanation path">
        {steps.slice(0, depth + 1).map((s, i) =>
          i > 0 && i < depth - 1 ? (
            i === 1 ? (
              <React.Fragment key="earlier">
                <span aria-hidden="true">›</span>
                <button
                  type="button"
                  onClick={() => onCrumb(depth - 2)}
                  title={steps
                    .slice(1, depth - 1)
                    .map((x) => x.calculation.title)
                    .join(" → ")}
                >
                  … {depth - 2} steps
                </button>
              </React.Fragment>
            ) : null
          ) : (
            <React.Fragment key={i}>
              {i > 0 && <span aria-hidden="true">›</span>}
              <button
                type="button"
                onClick={() => onCrumb(i)}
                title={s.calculation.title}
              >
                {i === 0 ? "Result" : s.calculation.title}
              </button>
            </React.Fragment>
          ),
        )}
      </nav>
      <div className="formula-tooltip-heading">
        {c.title}
        <small>{c.source || "Calculation from this run"}</small>
        <button
          type="button"
          className="formula-close"
          aria-label="Close explanations"
          onClick={onClose}
        >
          ×
        </button>
      </div>
      {c.summary && <p className="formula-summary">{c.summary}</p>}
      <pre>{c.formula}</pre>
      <div className="formula-tooltip-details">
        <dl>
          {c.variables.map((row, i) => (
            <div
              key={row.symbol + i}
              className={row.explain ? "formula-term-row" : ""}
              onMouseEnter={(e) => {
                if (row.explain)
                  onBranch(
                    depth,
                    row.symbol + i,
                    row.explain,
                    e.currentTarget,
                    true,
                  );
              }}
              onMouseLeave={onCancelOpen}
              onClick={(e) => {
                if (row.explain)
                  onBranch(
                    depth,
                    row.symbol + i,
                    row.explain,
                    e.currentTarget,
                    false,
                  );
              }}
            >
              <dt>
                {row.explain ? (
                  <button
                    type="button"
                    className="formula-term"
                    aria-label={`Explain ${row.label}`}
                    aria-expanded={steps[depth + 1]?.key === row.symbol + i}
                    onMouseEnter={(e) =>
                      onBranch(
                        depth,
                        row.symbol + i,
                        row.explain!,
                        e.currentTarget,
                        true,
                      )
                    }
                    onMouseLeave={onCancelOpen}
                    onFocus={(e) =>
                      onBranch(
                        depth,
                        row.symbol + i,
                        row.explain!,
                        e.currentTarget,
                        false,
                      )
                    }
                    onClick={(e) =>
                      onBranch(
                        depth,
                        row.symbol + i,
                        row.explain!,
                        e.currentTarget,
                        false,
                      )
                    }
                  >
                    <code>{row.symbol}</code>
                    <span>{row.label}</span>
                    <b aria-hidden="true">›</b>
                  </button>
                ) : (
                  <>
                    <code>{row.symbol}</code>
                    <span>{row.label}</span>
                  </>
                )}
              </dt>
              <dd>
                {numberText(row.value)}
                {row.unit && <small>{row.unit}</small>}
              </dd>
            </div>
          ))}
        </dl>
        {c.note && <p>{c.note}</p>}
        {c.links?.map((link) => (
          <a key={link.url} href={link.url} target="_blank" rel="noreferrer">
            {link.label} ↗
          </a>
        ))}
      </div>
      <div className="formula-result">
        <span>Result</span>
        <strong>
          {numberText(c.result)} <small>{c.unit}</small>
        </strong>
      </div>
      <div className="formula-tooltip-hint">
        {depth > 0 && (
          <button type="button" onClick={onBack}>
            ← Back
          </button>
        )}
        <span>
          {c.variables.some((v) => v.explain)
            ? "Hover or select a term to follow its derivation."
            : "Source reached."}{" "}
          Esc steps back.
        </span>
      </div>
    </div>
  );
}
