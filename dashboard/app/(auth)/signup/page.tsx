"use client";

import { Suspense, useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Eye, EyeIcon, EyeOffIcon, CheckCircle2, ShieldAlert } from "lucide-react";

export default function SignupPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-background"><p className="text-sm text-muted-foreground">Loading...</p></div>}>
      <SignupForm />
    </Suspense>
  );
}

function SignupForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [invite, setInvite] = useState<{
    id: string;
    org_id: string;
    role: string;
    org_name?: string;
  } | null>(null);
  const [validating, setValidating] = useState(true);
  const router = useRouter();
  const searchParams = useSearchParams();
  const code = searchParams.get("code");

  useEffect(() => {
    if (!code) {
      setValidating(false);
      return;
    }
    validateInvite(code);
  }, [code]);

  async function validateInvite(inviteCode: string) {
    const supabase = createClient();

    const { data, error } = await supabase
      .from("dashboard_invite")
      .select("id, org_id, role, expires_at, used_at")
      .eq("code", inviteCode)
      .single();

    if (error || !data) {
      setError("Invalid invite code.");
      setValidating(false);
      return;
    }

    if (data.used_at) {
      setError("This invite code has already been used.");
      setValidating(false);
      return;
    }

    if (new Date(data.expires_at) < new Date()) {
      setError("This invite code has expired.");
      setValidating(false);
      return;
    }

    // Get org name
    const { data: org } = await supabase
      .from("organization")
      .select("name")
      .eq("id", data.org_id)
      .single();

    setInvite({
      id: data.id,
      org_id: data.org_id,
      role: data.role,
      org_name: org?.name || "Unknown Organization",
    });
    setValidating(false);
  }

  async function handleSignup(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!invite) return;
    setError("");
    setLoading(true);

    const supabase = createClient();

    const { data: authData, error: authError } = await supabase.auth.signUp({
      email,
      password,
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    if (!authData.user) {
      setError("Signup failed.");
      setLoading(false);
      return;
    }

    // Update the member record with the email
    const { error: memberError } = await supabase
      .from("member")
      .update({ email, status: "active", updated_at: new Date().toISOString() })
      .eq("org_id", invite.org_id)
      .eq("email", email)
      .eq("status", "pending");

    // If no pending member with this email, create one
    if (memberError) {
      await supabase.from("member").insert({
        org_id: invite.org_id,
        role: invite.role,
        name: email.split("@")[0],
        email,
        status: "active",
      });
    }

    // Mark invite as used
    await supabase
      .from("dashboard_invite")
      .update({ used_at: new Date().toISOString() })
      .eq("id", invite.id);

    setSuccess(true);
    setLoading(false);
  }

  // No invite code provided
  if (!code && !validating) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
          <CardHeader className="flex flex-col items-center gap-2">
            <div className="relative flex size-[68px] shrink-0 items-center justify-center rounded-full backdrop-blur-xl md:size-24 before:absolute before:inset-0 before:rounded-full before:bg-gradient-to-b before:from-horus/30 before:to-transparent before:opacity-30">
              <div className="relative z-10 flex size-12 items-center justify-center rounded-full bg-background dark:bg-muted/80 shadow-xs ring-1 ring-inset ring-border md:size-16">
                <ShieldAlert className="size-6 text-muted-foreground md:size-8" />
              </div>
            </div>
            <div className="flex flex-col space-y-1.5 text-center">
              <CardTitle className="md:text-xl font-medium">
                Invite required
              </CardTitle>
              <CardDescription>
                Eyes of Horus uses invite-only registration. Contact your administrator to get an invite link.
              </CardDescription>
            </div>
          </CardHeader>
          <Separator />
          <Link href="/login">
            <Button variant="outline" className="w-full">
              Back to login
            </Button>
          </Link>
        </Card>
      </div>
    );
  }

  // Validating invite code
  if (validating) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <p className="text-sm text-muted-foreground">Validating invite...</p>
      </div>
    );
  }

  // Invalid/expired invite
  if (error && !invite) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
          <CardHeader className="flex flex-col items-center gap-2">
            <div className="flex flex-col space-y-1.5 text-center">
              <CardTitle className="md:text-xl font-medium">
                Invalid invite
              </CardTitle>
              <CardDescription>{error}</CardDescription>
            </div>
          </CardHeader>
          <Separator />
          <Link href="/login">
            <Button variant="outline" className="w-full">
              Back to login
            </Button>
          </Link>
        </Card>
      </div>
    );
  }

  // Success
  if (success) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
          <CardHeader className="flex flex-col items-center gap-4">
            <div className="relative flex size-[68px] shrink-0 items-center justify-center rounded-full backdrop-blur-xl md:size-24 before:absolute before:inset-0 before:rounded-full before:bg-gradient-to-b before:from-green-500/30 before:to-transparent before:opacity-30">
              <div className="relative z-10 flex size-12 items-center justify-center rounded-full bg-background dark:bg-muted/80 shadow-xs ring-1 ring-inset ring-border md:size-16">
                <CheckCircle2 className="size-6 text-green-500 md:size-8" />
              </div>
            </div>
            <div className="flex flex-col space-y-1.5 text-center">
              <CardTitle className="md:text-xl font-medium">Check your email</CardTitle>
              <CardDescription>
                We sent a confirmation link to <strong className="text-foreground">{email}</strong>.
                Click it to activate your account.
              </CardDescription>
            </div>
          </CardHeader>
          <Separator />
          <Link href="/login">
            <Button variant="outline" className="w-full">Back to login</Button>
          </Link>
        </Card>
      </div>
    );
  }

  // Valid invite — show signup form
  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
        <CardHeader className="flex flex-col items-center gap-2">
          <div className="relative flex size-[68px] shrink-0 items-center justify-center rounded-full backdrop-blur-xl md:size-24 before:absolute before:inset-0 before:rounded-full before:bg-gradient-to-b before:from-horus/30 before:to-transparent before:opacity-30">
            <div className="relative z-10 flex size-12 items-center justify-center rounded-full bg-background dark:bg-muted/80 shadow-xs ring-1 ring-inset ring-border md:size-16">
              <Eye className="size-6 text-horus md:size-8" />
            </div>
          </div>
          <div className="flex flex-col space-y-1.5 text-center">
            <p className="font-mono text-[10px] tracking-[0.3em] uppercase text-horus">
              eyes of horus
            </p>
            <CardTitle className="md:text-xl font-medium">
              Join {invite?.org_name}
            </CardTitle>
            <CardDescription>
              You have been invited as <strong className="text-foreground">{invite?.role}</strong>. Create your account to get started.
            </CardDescription>
          </div>
        </CardHeader>

        <Separator />

        <CardContent className="p-0">
          <form onSubmit={handleSignup} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2.5">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@company.com"
                className="rounded-lg"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
            </div>

            <div className="flex flex-col gap-2.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  className="pe-9 rounded-lg"
                  placeholder="Min 6 characters"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                />
                <button
                  className="text-muted-foreground/80 hover:text-foreground absolute inset-y-0 end-0 flex h-full w-9 items-center justify-center rounded-e-md transition-colors outline-none"
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOffIcon size={16} /> : <EyeIcon size={16} />}
                </button>
              </div>
            </div>

            {error && <p className="text-sm text-destructive">{error}</p>}

            <Button
              type="submit"
              className="w-full bg-horus text-background hover:bg-horus/90"
              disabled={loading}
            >
              {loading ? "Creating account..." : "Create account"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
