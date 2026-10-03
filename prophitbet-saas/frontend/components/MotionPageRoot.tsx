"use client";

type Props = {
  children: React.ReactNode;
  className?: string;
  reviveKey?: string;
};

export default function MotionPageRoot({ children, className }: Props) {
  return <div className={className}>{children}</div>;
}
