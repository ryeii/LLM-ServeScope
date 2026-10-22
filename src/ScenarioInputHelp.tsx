import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { inputExplanation, type InputHelpContext } from "./inputHelp";

type Control = HTMLInputElement | HTMLSelectElement | HTMLElement;
type Active = {
  element: Control;
  label: string;
  value: string;
  unit: string;
  anchor: DOMRect;
  note: string;
};
const selector =
  '.sidebar input:not([type="file"]), .sidebar select, .sidebar [data-input-help], .native-form input, .native-form select';

function snapshot(element: Control): Active {
  const label =
    element.getAttribute("data-input-help") ||
    element.getAttribute("aria-label") ||
    "";
  const value =
    element instanceof HTMLSelectElement
      ? element.selectedOptions[0]?.textContent?.trim() || element.value
      : element instanceof HTMLInputElement
        ? element.type === "checkbox"
          ? element.checked
            ? "Enabled"
            : "Disabled"
          : element.value || "Editing…"
        : element.textContent?.trim() || "";
  return {
    element,
    label,
    value,
    anchor: element.getBoundingClientRect(),
    unit: element.closest("label")?.querySelector("small")?.textContent || "",
    note: element.closest("label")?.getAttribute("data-input-tip") || "",
  };
}

/** Delegated help leaves native input semantics, editing and keyboard shortcuts intact. */
export default function ScenarioInputHelp(context: InputHelpContext) {
  const [active, setActive] = useState<Active | null>(null);
  const current = useRef<Active | null>(null);
  const pending = useRef<Control | null>(null);
  const popup = useRef<HTMLElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const id = useId();
  const close = () => {
    clearTimeout(timer.current);
    pending.current = null;
    current.current = null;
    setActive(null);
  };
  const show = (element: Control) => {
    clearTimeout(timer.current);
    pending.current = null;
    const next = snapshot(element);
    if (!next.label) return;
    current.current = next;
    setActive(next);
  };
  useEffect(() => {
    const control = (target: EventTarget | null) =>
      target instanceof Element ? target.closest<Control>(selector) : null;
    const enter = (event: MouseEvent) => {
      const element = control(event.target);
      if (!element) return;
      if (current.current?.element === element) {
        clearTimeout(timer.current);
        return;
      }
      clearTimeout(timer.current);
      pending.current = element;
      timer.current = setTimeout(() => {
        if (element.isConnected) show(element);
      }, 220);
    };
    const leave = (event: MouseEvent) => {
      const element = control(event.target);
      if (
        !element ||
        (event.relatedTarget instanceof Node &&
          element.contains(event.relatedTarget))
      )
        return;
      if (popup.current?.contains(event.relatedTarget as Node)) {
        clearTimeout(timer.current);
        return;
      }
      clearTimeout(timer.current);
      if (pending.current === element) pending.current = null;
      if (document.activeElement !== element)
        timer.current = setTimeout(close, 180);
    };
    const focus = (event: FocusEvent) => {
      const element = control(event.target);
      if (element) show(element);
      else if (!popup.current?.contains(event.target as Node)) close();
    };
    const blur = (event: FocusEvent) => {
      if (!popup.current?.contains(event.relatedTarget as Node)) close();
    };
    const edit = (event: Event) => {
      const element = control(event.target);
      if (element && current.current?.element === element) show(element);
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    const pointer = (event: PointerEvent) => {
      if (
        !current.current?.element.contains(event.target as Node) &&
        !popup.current?.contains(event.target as Node)
      )
        close();
    };
    const scroll = (event: Event) => {
      if (popup.current?.contains(event.target as Node)) return;
      const element = pending.current || current.current?.element;
      // Browser scroll-into-view can dispatch after mouseover/focus. Let that
      // movement settle, then anchor to the current rect rather than lose help.
      const rect = element?.getBoundingClientRect();
      if (
        element &&
        rect &&
        rect.bottom > 0 &&
        rect.top < window.innerHeight &&
        (element.matches(":hover") || document.activeElement === element)
      ) {
        clearTimeout(timer.current);
        pending.current = element;
        current.current = null;
        setActive(null);
        timer.current = setTimeout(() => {
          if (element.isConnected) show(element);
        }, 220);
      } else close();
    };
    document.addEventListener("mouseover", enter);
    document.addEventListener("mouseout", leave);
    document.addEventListener("focusin", focus);
    document.addEventListener("focusout", blur);
    document.addEventListener("input", edit);
    document.addEventListener("change", edit);
    document.addEventListener("keydown", key);
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", close);
    window.addEventListener("blur", close);
    return () => {
      clearTimeout(timer.current);
      document.removeEventListener("mouseover", enter);
      document.removeEventListener("mouseout", leave);
      document.removeEventListener("focusin", focus);
      document.removeEventListener("focusout", blur);
      document.removeEventListener("input", edit);
      document.removeEventListener("change", edit);
      document.removeEventListener("keydown", key);
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("blur", close);
    };
  }, []);

  // Preset changes and the pool editor can replace values without a DOM input event.
  useLayoutEffect(() => {
    if (!active) return;
    if (
      !active.element.isConnected ||
      !active.element.getClientRects().length
    ) {
      close();
      return;
    }
    const next = snapshot(active.element);
    if (next.value !== active.value || next.label !== active.label) {
      current.current = next;
      setActive(next);
    }
  }, [context.config, context.phase, active]);
  useEffect(() => {
    if (!active) return;
    const element = active.element;
    const previous = element.getAttribute("aria-describedby");
    element.setAttribute(
      "aria-describedby",
      [previous, id].filter(Boolean).join(" "),
    );
    return () => {
      if (previous) element.setAttribute("aria-describedby", previous);
      else element.removeAttribute("aria-describedby");
    };
  }, [active?.element, id]);

  if (!active) return null;
  const help = inputExplanation(active.label, context);
  const width = Math.min(410, window.innerWidth - 24);
  const rightFits = active.anchor.right + 12 + width <= window.innerWidth - 12;
  const left = rightFits
    ? active.anchor.right + 12
    : Math.max(
        12,
        Math.min(active.anchor.left, window.innerWidth - width - 12),
      );
  // On narrow screens, place help above/below the control rather than over it.
  const below = Math.max(0, window.innerHeight - active.anchor.bottom - 24);
  const above = Math.max(0, active.anchor.top - 24);
  const placeBelow = below >= Math.min(390, above);
  const height = rightFits
    ? Math.min(480, window.innerHeight - 24)
    : Math.min(390, Math.max(96, placeBelow ? below : above));
  const top = rightFits
    ? Math.max(
        12,
        Math.min(active.anchor.top, window.innerHeight - height - 12),
      )
    : placeBelow
      ? active.anchor.bottom + 12
      : Math.max(12, active.anchor.top - height - 12);
  return createPortal(
    <section
      ref={popup}
      id={id}
      role="tooltip"
      className="scenario-input-help"
      data-input-help-label={active.label}
      style={{ left, top, width, maxHeight: height }}
      onMouseEnter={() => clearTimeout(timer.current)}
      onMouseLeave={() => {
        if (document.activeElement !== active.element) close();
      }}
    >
      <h3>{help.title}</h3>
      <div className="input-help-value">
        <span>Current input</span>
        <strong>
          {active.value}
          {active.unit ? " " + active.unit : ""}
        </strong>
      </div>
      <p>{help.meaning}</p>
      <h4>What it changes</h4>
      <p>{help.effect}</p>
      {active.note && <p>{active.note}</p>}
      <h4>Evidence & assumptions</h4>
      <p>{help.evidence}</p>
      <div className="input-help-hint">
        Editable inputs explain the current scenario. Results update after Run
        scenario. Esc closes this explanation.
      </div>
    </section>,
    document.body,
  );
}
