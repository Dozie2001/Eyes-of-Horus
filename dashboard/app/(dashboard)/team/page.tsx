"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { UserPlus, Copy, Check, Shield, Eye as EyeIcon, Crown, UserX, RotateCcw } from "lucide-react";
import { toast } from "sonner";

function generateInviteCode(): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "INV-";
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

interface Member {
  id: string;
  org_id: string;
  role: string;
  name: string;
  email: string | null;
  telegram_chat_id: string | null;
  telegram_username: string | null;
  status: string;
  created_at: string;
  updated_at: string;
}

const roleBadge: Record<string, { color: string; icon: typeof Shield }> = {
  guard: { color: "text-blue-400 bg-blue-400/10", icon: Shield },
  supervisor: { color: "text-yellow-400 bg-yellow-400/10", icon: EyeIcon },
  admin: { color: "text-purple-400 bg-purple-400/10", icon: Crown },
  owner: { color: "text-horus bg-horus/10", icon: Crown },
};

export default function TeamPage() {
  const supabase = createClient();
  const queryClient = useQueryClient();

  const [inviteName, setInviteName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteTelegram, setInviteTelegram] = useState("");
  const [inviteRole, setInviteRole] = useState("guard");
  const [copiedLink, setCopiedLink] = useState("");

  const { data: members = [], isLoading } = useQuery({
    queryKey: ["team-members"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("member")
        .select("*")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
    refetchInterval: 15_000,
  });

  const inviteMutation = useMutation({
    mutationFn: async () => {
      const code = generateInviteCode();
      const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();

      // Get user's org_id
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("Not authenticated");

      const { data: membership } = await supabase
        .from("member")
        .select("org_id")
        .eq("email", user.email)
        .eq("status", "active")
        .single();

      if (!membership) throw new Error("No organization found");

      // Create member record (pending)
      const { data: member, error: memberError } = await supabase
        .from("member")
        .insert({
          org_id: membership.org_id,
          role: inviteRole,
          name: inviteName,
          email: inviteEmail || null,
          telegram_username: inviteTelegram || null,
          status: "pending",
        })
        .select("id")
        .single();

      if (memberError) throw memberError;

      // Create invite code
      const { error: inviteError } = await supabase
        .from("dashboard_invite")
        .insert({
          org_id: membership.org_id,
          role: inviteRole,
          code,
          created_by: member.id,
          expires_at: expiresAt,
        });

      if (inviteError) throw inviteError;

      return { code, role: inviteRole, name: inviteName };
    },
    onSuccess: (data) => {
      const link = `${window.location.origin}/signup?code=${data.code}`;
      setCopiedLink(link);
      toast.success(`Invite created for ${data.name} (${data.role})`);
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
      setInviteName("");
      setInviteEmail("");
      setInviteTelegram("");
    },
    onError: (err) => {
      toast.error(`Failed to create invite: ${(err as Error).message}`);
    },
  });

  const revokeMutation = useMutation({
    mutationFn: async (memberId: string) => {
      const { error } = await supabase
        .from("member")
        .update({ status: "revoked", updated_at: new Date().toISOString() })
        .eq("id", memberId);
      if (error) throw error;
    },
    onSuccess: () => {
      toast.success("Member revoked");
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
    },
  });

  const restoreMutation = useMutation({
    mutationFn: async (memberId: string) => {
      const { error } = await supabase
        .from("member")
        .update({ status: "active", updated_at: new Date().toISOString() })
        .eq("id", memberId);
      if (error) throw error;
    },
    onSuccess: () => {
      toast.success("Member restored");
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
    },
  });

  async function copyLink() {
    await navigator.clipboard.writeText(copiedLink);
    toast.success("Invite link copied");
  }

  const active = members.filter((m: Member) => m.status === "active");
  const pending = members.filter((m: Member) => m.status === "pending");
  const revoked = members.filter((m: Member) => m.status === "revoked");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Team</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage who has access to this location.
        </p>
      </div>

      {/* Invite form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <UserPlus className="size-4" />
            Invite team member
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                placeholder="Chidi Okafor"
                value={inviteName}
                onChange={(e) => setInviteName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="role">Role</Label>
              <select
                id="role"
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="guard">Guard (Telegram only)</option>
                <option value="supervisor">Supervisor (Telegram + Dashboard read-only)</option>
                <option value="admin">Admin (Full dashboard access)</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">
                Email {inviteRole === "guard" ? "(optional)" : "(required for dashboard)"}
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="chidi@company.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="telegram">Telegram username (optional)</Label>
              <Input
                id="telegram"
                placeholder="@chidi_security"
                value={inviteTelegram}
                onChange={(e) => setInviteTelegram(e.target.value)}
              />
            </div>
          </div>

          <Button
            onClick={() => inviteMutation.mutate()}
            disabled={!inviteName || inviteMutation.isPending}
            className="bg-horus text-background hover:bg-horus/90"
          >
            {inviteMutation.isPending ? "Creating..." : "Create invite"}
          </Button>

          {copiedLink && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50 border">
              <code className="text-xs flex-1 truncate">{copiedLink}</code>
              <Button variant="ghost" size="sm" onClick={copyLink}>
                <Copy className="size-3.5" />
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active members */}
      <div className="space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground">
          Active ({active.length})
        </h2>
        {active.map((m: Member) => {
          const badge = roleBadge[m.role] || roleBadge.guard;
          const Icon = badge.icon;
          return (
            <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border bg-card">
              <div className="flex items-center gap-3">
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${badge.color}`}>
                  <Icon className="size-3" />
                  {m.role}
                </span>
                <div>
                  <p className="text-sm font-medium">{m.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {[m.email, m.telegram_username].filter(Boolean).join(" / ") || "No contact info"}
                  </p>
                </div>
              </div>
              {m.role !== "owner" && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive"
                  onClick={() => revokeMutation.mutate(m.id)}
                >
                  <UserX className="size-3.5 mr-1" />
                  Revoke
                </Button>
              )}
            </div>
          );
        })}
      </div>

      {/* Pending */}
      {pending.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-muted-foreground">
            Pending ({pending.length})
          </h2>
          {pending.map((m: Member) => {
            const badge = roleBadge[m.role] || roleBadge.guard;
            const Icon = badge.icon;
            return (
              <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border bg-card opacity-60">
                <div className="flex items-center gap-3">
                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${badge.color}`}>
                    <Icon className="size-3" />
                    {m.role}
                  </span>
                  <div>
                    <p className="text-sm font-medium">{m.name}</p>
                    <p className="text-xs text-muted-foreground">Awaiting signup</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Revoked */}
      {revoked.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-muted-foreground">
            Removed ({revoked.length})
          </h2>
          {revoked.map((m: Member) => (
            <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border bg-card opacity-40">
              <div>
                <p className="text-sm font-medium">{m.name}</p>
                <p className="text-xs text-muted-foreground">
                  Was {m.role} / {m.email || m.telegram_username || "no contact"}
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => restoreMutation.mutate(m.id)}
              >
                <RotateCcw className="size-3.5 mr-1" />
                Restore
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
