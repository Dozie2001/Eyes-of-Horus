import { CameraList } from "@/components/cameras/camera-list";

export default function CamerasPage() {
  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-xl font-light tracking-tight">Cameras</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Camera status and profile management
        </p>
      </div>
      <CameraList />
    </div>
  );
}
