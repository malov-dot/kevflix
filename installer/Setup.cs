using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using Microsoft.Win32;

// Per-user installer. It never edits Plex, services, media folders or other applications.
// Code/runtime go to the install folder; the user's settings and data live in the KevFLIX profile
// (%LOCALAPPDATA%\KevFLIX\profile by default) and are never replaced by an install or update.
class Setup : Form {
    const string Product = "KevFLIX", Version = BuildInfo.Version;
    static string Root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", Product);
    static bool Silent, Updating, Startup = true, VoxInPlace;
    static int WaitPid;
    static string VoxModel, VoxRuntime, VoxNote;
    static bool VoxOk;
    Label status; Button install; CheckBox startup, inPlace; TextBox modelBox, runtimeBox;
    Setup() {
        Text="KevFLIX Setup " + Version; Width=700; Height=520; StartPosition=FormStartPosition.CenterScreen;
        Font=new Font("Segoe UI",10); FormBorderStyle=FormBorderStyle.FixedDialog; MaximizeBox=false;
        var title=new Label {Text="Install KevFLIX PC",Left=28,Top=20,Width=600,Height=38,Font=new Font("Segoe UI",20)};
        status=new Label {Text="Installs the desktop control panel and private Python runtime.\nYour Plex account, settings and library stay in your KevFLIX profile and are kept on update.\n\nDevelopment build: playback and installation have not been tested.",Left=30,Top=65,Width=625,Height=95};
        // Vox actor engine: per-machine, private source locations (never written into the package or the example).
        var voxTitle=new Label {Text="Vox actor voice engine (needed for dubbed voices)",Left=30,Top=165,Width=620,Height=24,Font=new Font("Segoe UI",10,FontStyle.Bold)};
        var modelLabel=new Label {Text="Vox voice model file (voxcpm2-actor-q4_k.gguf):",Left=30,Top=192,Width=620,Height=22};
        modelBox=new TextBox {Left=30,Top=216,Width=520};
        var modelBrowse=new Button {Text="Browse...",Left=560,Top=214,Width=95};
        var runtimeLabel=new Label {Text="Vox native runtime folder (audiocpp.dll and its CUDA/ggml DLLs):",Left=30,Top=248,Width=620,Height=22};
        runtimeBox=new TextBox {Left=30,Top=272,Width=520};
        var runtimeBrowse=new Button {Text="Browse...",Left=560,Top=270,Width=95};
        inPlace=new CheckBox {Text="Use these files where they are (do not copy them into the profile)",Left=30,Top=304,Width=600};
        string preModel, preRuntime; ConfiguredVox(out preModel, out preRuntime);
        if(preModel!=null)modelBox.Text=preModel; if(preRuntime!=null)runtimeBox.Text=preRuntime;
        modelBrowse.Click+=delegate {using(var d=new OpenFileDialog{Filter="Vox model (*.gguf)|*.gguf",Title="Vox voice model"}){if(d.ShowDialog(this)==DialogResult.OK)modelBox.Text=d.FileName;}};
        runtimeBrowse.Click+=delegate {using(var d=new FolderBrowserDialog{Description="Folder containing audiocpp.dll"}){if(d.ShowDialog(this)==DialogResult.OK)runtimeBox.Text=d.SelectedPath;}};
        startup=new CheckBox {Text="Start KevFLIX with Windows (new installs; an update keeps your choice)",Checked=true,Left=30,Top=340,Width=600};
        install=new Button {Text="Install",Left=520,Top=430,Width=135};
        bool finished=false;
        install.Click+=async delegate {
            if(finished){Close();return;}
            Startup=startup.Checked;
            VoxModel=modelBox.Text.Trim().Length>0?modelBox.Text.Trim():null;
            VoxRuntime=runtimeBox.Text.Trim().Length>0?runtimeBox.Text.Trim():null;
            VoxInPlace=inPlace.Checked;
            foreach(Control c in Controls)if(c!=status)c.Enabled=false;
            try {
                await Task.Run(()=>Install());
                status.Text=(VoxOk?"Installed. KevFLIX PC is opening.\nConnect Plex or choose movie folders, then pair your TV.\n":"Installed WITHOUT a usable Vox engine. Dubbed voices are unavailable until it is provisioned (run KevFLIX Setup again with the model file and runtime folder).\n")+(VoxNote??"");
            } catch(Exception ex){status.Text="Installation did not complete: "+ex.Message;}
            finally{finished=true;install.Text="Close";install.Enabled=true;}
        };
        Controls.AddRange(new Control[]{title,status,voxTitle,modelLabel,modelBox,modelBrowse,runtimeLabel,runtimeBox,runtimeBrowse,inPlace,startup,install});
    }
    static string ProfileRoot() {
        string env=Environment.GetEnvironmentVariable("KEVFLIX_PROFILE");
        if(!string.IsNullOrEmpty(env))return env;
        string local=Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string pointer=Path.Combine(local,"KevFLIX","profile-location.json");
        if(File.Exists(pointer)){var m=System.Text.RegularExpressions.Regex.Match(File.ReadAllText(pointer),"\"profile_root\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");if(m.Success)return Unescape(m.Groups[1].Value);}
        return Path.Combine(local,"KevFLIX","profile");
    }
    static string Unescape(string s){return s.Replace("\\\\","\u0001").Replace("\\/","/").Replace("\u0001","\\");}
    static string JsonString(string text,string key) {
        var m=System.Text.RegularExpressions.Regex.Match(text,"\""+key+"\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
        return m.Success?Unescape(m.Groups[1].Value):null;
    }
    // Pre-fill from an existing configured Vox install in the user's profile (read only; nothing is changed here).
    static void ConfiguredVox(out string model, out string runtime) {
        model=null;runtime=null;
        try {
            string config=Path.Combine(ProfileRoot(),"relay-config.json");
            if(!File.Exists(config))return;
            string manifest=JsonString(File.ReadAllText(config),"actor_runtime_manifest");
            if(string.IsNullOrEmpty(manifest)||!File.Exists(manifest))return;
            string text=File.ReadAllText(manifest);
            string m=JsonString(text,"model"), lib=JsonString(text,"library");
            if(m!=null&&File.Exists(m))model=m;
            if(lib!=null&&File.Exists(lib))runtime=Path.GetDirectoryName(lib);
        } catch {}
    }
    [STAThread] static int Main(string[] args) {
        try {
            for(int i=0;i<args.Length;i++) {
                if(args[i]=="/silent") Silent=true;
                else if(args[i]=="/update") Updating=true;
                else if(args[i]=="/target") Root=Path.GetFullPath(args[++i]);
                else if(args[i]=="/wait-pid") WaitPid=int.Parse(args[++i]);
                else if(args[i]=="/vox-model") VoxModel=Path.GetFullPath(args[++i]);
                else if(args[i]=="/vox-runtime") VoxRuntime=Path.GetFullPath(args[++i]);
                else if(args[i]=="/vox-in-place") VoxInPlace=true;
                else throw new Exception("Unknown setup option");
            }
            using(var mutex=new Mutex(false,"Local\\KevFLIXInstaller")) {
                if(!mutex.WaitOne(0)) throw new Exception("Another KevFLIX installation is already running.");
                try {if(Silent){Install();return 0;} Application.EnableVisualStyles();Application.Run(new Setup());return 0;}
                finally{mutex.ReleaseMutex();}
            }
        } catch(Exception ex){if(!Silent)MessageBox.Show(ex.Message,"KevFLIX setup");return 1;}
    }
    static string Safe(string relative) {
        string prefix=Path.GetFullPath(Root).TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar;
        string path=Path.GetFullPath(Path.Combine(prefix,relative));
        if(!path.StartsWith(prefix,StringComparison.OrdinalIgnoreCase)||relative.Contains(":")) throw new Exception("Unsafe package path");
        for(var parent=new DirectoryInfo(Path.GetDirectoryName(path));parent!=null;parent=parent.Parent) {
            if(parent.Exists && (parent.Attributes & FileAttributes.ReparsePoint)!=0) throw new Exception("Installation through linked directories is not supported");
        }
        if(File.Exists(path)&&(File.GetAttributes(path)&FileAttributes.ReparsePoint)!=0)throw new Exception("Linked installation file is not supported");
        return path;
    }
    static void Install() {
        string marker=Path.Combine(Root,"kevflix-install.txt");
        bool existing=File.Exists(marker)&&File.ReadAllText(marker).StartsWith("KevFLIX\n",StringComparison.Ordinal);
        if(Updating&&!existing)throw new Exception("Automatic updates require an installer-managed KevFLIX directory");
        if(Directory.Exists(Root)&&Directory.EnumerateFileSystemEntries(Root).Any()&&!existing)throw new Exception("This folder contains an unrelated installation. It will not be overwritten.");
        if(WaitPid>0) {try{var process=Process.GetProcessById(WaitPid);if(!process.WaitForExit(60000))throw new Exception("KevFLIX is still running; update deferred");}catch(ArgumentException){}}
        // An ordinary reinstall never kills a running worker. Locked files cause rollback.
        Directory.CreateDirectory(Root);
        Directory.CreateDirectory(Safe("state"));
        var instanceLock=new FileStream(Safe("state/worker.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);
        // Held by a running app guard (relay/app_runtime/guard.py): an install never replaces files under it.
        var guardLock=new FileStream(Safe("state/install-guard.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);
        string previousMarker=existing?File.ReadAllText(marker):null;
        string oldRequirements=File.Exists(Safe("requirements.txt"))?File.ReadAllText(Safe("requirements.txt")):null;
        var changed=new List<string>();var created=new List<string>();var removed=new List<string>();
        // This build's files (pre-pass over the payload) and the files a PREVIOUS build provably shipped: path +
        // sha256 from the previous install's app-files.txt, or, for builds that predate it, this payload's
        // legacy-shipped.txt (the enumerated files of the earlier published package). Ownership is never inferred
        // from a folder; an obsolete file is removed only if BOTH its path and its sha256 match.
        var newFiles=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var owned=new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
        using(var payload=Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip"))
        using(var zip=new ZipArchive(payload,ZipArchiveMode.Read)) {
            foreach(var entry in zip.Entries){string n=entry.FullName.Replace('\\','/');if(!n.EndsWith("/"))newFiles.Add(n);}
            if(existing&&!File.Exists(Safe("app-files.txt"))) {
                var legacy=zip.GetEntry("legacy-shipped.txt");
                if(legacy!=null)using(var reader=new StreamReader(legacy.Open()))ReadOwned(reader.ReadToEnd(),owned);
            }
        }
        if(existing&&File.Exists(Safe("app-files.txt")))ReadOwned(File.ReadAllText(Safe("app-files.txt")),owned);
        // Preflight (before anything changes): an unknown or locally changed file in a folder the integrity check
        // covers would make every start refuse. Such a file is never deleted: the install stops and names it.
        var obsolete=new List<string>();var conflicts=new List<string>();
        foreach(var top in new[]{"relay","actor-tts","web","tv","scripts","runtime"}) {
            string dir=Safe(top);
            if(!Directory.Exists(dir))continue;
            foreach(var path in Directory.EnumerateFiles(dir,"*",SearchOption.AllDirectories)) {
                string name=Rel(path);
                if(newFiles.Contains(name)||name.Contains("/__pycache__/")||name.EndsWith(".pyc")||name.EndsWith(".pyo"))continue;
                string want;
                if(owned.TryGetValue(name,out want)&&Sha256(path)==want)obsolete.Add(name);else conflicts.Add(name);
            }
        }
        string conflictError=conflicts.Count==0?null:
            "These files in KevFLIX's code folders were not shipped by a KevFLIX build (or were changed), so they "+
            "are kept and the installation stopped before changing anything: "+string.Join(", ",conflicts.Take(10))+
            (conflicts.Count>10?" ...":"")+". Move them out of "+Root+" and run Setup again.";
        // Owned files outside the covered folders (e.g. retired docs): removed only with a matching hash.
        foreach(var item in owned) {
            if(newFiles.Contains(item.Key)||obsolete.Contains(item.Key))continue;
            string top=item.Key.Split('/')[0];
            if(new[]{"relay","actor-tts","web","tv","scripts","runtime","state","models",".venv","setup-backups"}.Contains(top)||item.Key=="relay-config.json")continue;
            string path=Safe(item.Key);
            if(File.Exists(path)&&Sha256(path)==item.Value)obsolete.Add(item.Key);
        }
        string backupName="setup-backups/"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N");
        string backup=Safe(backupName);Directory.CreateDirectory(backup);
        bool success=false, replacedEnvironment=false;
        try {
            if(conflictError!=null)throw new Exception(conflictError);     // nothing has been changed yet
            if(!existing)File.WriteAllText(marker,"KevFLIX\npartial\n");
            using(var payload=Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip"))
            using(var zip=new ZipArchive(payload,ZipArchiveMode.Read)) {
                foreach(var entry in zip.Entries) {
                    string name=entry.FullName.Replace('\\','/');
                    if(name.EndsWith("/"))continue;
                    if(name.StartsWith("state/")||name.StartsWith("models/")||name=="relay-config.json"||name.StartsWith("private-activation/"))throw new Exception("Private data in package");
                    string dest=Safe(name);Directory.CreateDirectory(Path.GetDirectoryName(dest));
                    if(File.Exists(dest)){string old=Path.Combine(backup,name);Directory.CreateDirectory(Path.GetDirectoryName(old));File.Copy(dest,old);changed.Add(name);}else created.Add(name);
                    using(var input=entry.Open())using(var output=File.Create(dest)){input.CopyTo(output);}
                }
            }
            // Obsolete files with PROVEN ownership (path + sha256 from the preflight). Each is backed up at
            // backup/<its original path> first; rollback restores it to that original path.
            foreach(var name in obsolete) {
                string path=Safe(name);
                if(!File.Exists(path))continue;
                string old=Path.Combine(backup,name);Directory.CreateDirectory(Path.GetDirectoryName(old));
                File.Copy(path,old,true);File.Delete(path);removed.Add(name);
            }
            // Always run: it keeps existing settings and the Windows-startup preference, migrates an old in-install
            // config by copy and creates settings only on a brand-new PC. It does NOT change app-owned fields; those
            // are committed as the very last required step below.
            Run(Path.Combine(Root,"runtime/python/python.exe"),"\""+Path.Combine(Root,"scripts/install_pc.py")+"\" --startup "+(Startup?"on":"off"),true);
            if(existing && Directory.Exists(Safe(".venv")) && oldRequirements!=File.ReadAllText(Safe("requirements.txt"))) {
                Directory.Move(Safe(".venv"),Safe(backupName+"/previous-venv"));replacedEnvironment=true;
                Run(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell/v1.0/powershell.exe"),
                    "-NoProfile -ExecutionPolicy Bypass -File \""+Safe("Setup-Worker.ps1")+"\"",true);
            }
            File.WriteAllText(marker,"KevFLIX\n"+Version+"\n"+BuildInfo.Revision+"\n");
            using(var key=Registry.CurrentUser.CreateSubKey("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\KevFLIX")) {
                key.SetValue("DisplayName",Product);key.SetValue("DisplayVersion",Version);key.SetValue("Publisher",Product);key.SetValue("InstallLocation",Root);
                key.SetValue("UninstallString","\""+Path.Combine(Root,"runtime/python/pythonw.exe")+"\" \""+Path.Combine(Root,"scripts/uninstall_pc.py")+"\"");key.SetValue("NoModify",1);key.SetValue("NoRepair",1);
            }
            // LAST required step: commit the app-owned settings fields of THIS build (Vox adapter pins, bundled lease
            // client) in one atomic write of the profile settings. If it fails, nothing was written and the code is
            // rolled back below, so old code never sees new pins. Nothing after this point can roll the code back.
            Run(Path.Combine(Root,"runtime/python/python.exe"),"\""+Path.Combine(Root,"scripts/install_pc.py")+"\" --commit-app-pins",true);
            success=true;
            try{WriteProfileState("desktop-update-result.json","{\"version\":\""+Version+"\",\"revision\":\""+BuildInfo.Revision+"\",\"ok\":true}");}catch{}
            // The applied update's pointer (relay/desktop_updates.py keeps downloads in the profile).
            try{string pending=Path.Combine(ProfileRoot(),"desktop-updates","pending.json");if(File.Exists(pending))File.Delete(pending);}catch{}
            try {
                // Vox provisioning (GUI pickers or /vox-model /vox-runtime): the hash-verified provisioner checks the
                // model sha256/size and every approved runtime artifact, copies or references them, then writes
                // actor_runtime_manifest and the Vox block into the user's profile settings (nothing else). With no
                // sources it only verifies an already configured install. A failure never rolls back the app; it is
                // shown with the provisioner's exact message, and success is never reported when Vox is unusable.
                // Order: 1) the engine's Python environment (the existing Setup-Worker flow: .venv from the bundled
                // runtime + requirements.txt, i.e. numpy, soundfile and the rest), 2) Vox provisioning.
                string depsError=null;
                if((VoxModel!=null||VoxRuntime!=null)&&!File.Exists(Safe(".venv/Scripts/python.exe"))) {
                    string depsOut;
                    int depsCode=RunCapture(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell/v1.0/powershell.exe"),
                        "-NoProfile -ExecutionPolicy Bypass -File \""+Safe("Setup-Worker.ps1")+"\"",out depsOut);
                    if(depsCode!=0){
                        string tail=(depsOut??"").Trim();if(tail.Length>400)tail=tail.Substring(tail.Length-400);
                        depsError="the Python dependencies for the Vox engine could not be installed (Setup-Worker.ps1 exit "+depsCode+"): "+tail;
                    }
                }
                string voxArgs="\""+Path.Combine(Root,"scripts/provision_vox.py")+"\"";
                if(VoxModel!=null)voxArgs+=" --model-source \""+VoxModel+"\"";
                if(VoxRuntime!=null)voxArgs+=" --runtime-source \""+VoxRuntime+"\"";
                if(VoxInPlace)voxArgs+=" --reference-in-place";
                string voxOut;
                int voxCode=RunCapture(Path.Combine(Root,"runtime/python/python.exe"),voxArgs,out voxOut);
                VoxOk=voxCode==0&&depsError==null;
                string detail=JsonString(voxOut??"","error");
                VoxNote=VoxOk?"Vox actor engine verified and configured.":"Vox actor engine not usable: "+
                    (depsError!=null?depsError+(detail!=null?"; "+detail:""):(detail??("provisioner exit "+voxCode)));
                WriteProfileState("vox-status.txt",(VoxOk?"ok\n":"not usable\n")+VoxNote+"\n");
            } catch(Exception voxError) {
                // after the commit point: reported, never rolls the installed code back
                VoxOk=false;VoxNote="Vox actor engine not usable: "+voxError.Message;
                try{WriteProfileState("vox-status.txt","not usable\n"+VoxNote+"\n");}catch{}
            }
        } catch {
            if(success)throw;                               // (not reached: post-commit steps never throw)
            if(replacedEnvironment){if(Directory.Exists(Safe(".venv")))Directory.Move(Safe(".venv"),Safe(backupName+"/failed-venv"));Directory.Move(Safe(backupName+"/previous-venv"),Safe(".venv"));}
            foreach(var name in changed)File.Copy(Path.Combine(backup,name),Safe(name),true);
            // removed files were backed up at backup/<original path>: each is restored to its ORIGINAL path
            foreach(var name in removed){string dest=Safe(name);Directory.CreateDirectory(Path.GetDirectoryName(dest));File.Copy(Path.Combine(backup,name),dest,true);}
            foreach(var name in created)if(File.Exists(Safe(name)))File.Delete(Safe(name));
            if(previousMarker!=null)File.WriteAllText(marker,previousMarker);else File.WriteAllText(marker,"KevFLIX\npartial\n");
            try{using(var key=Registry.CurrentUser.OpenSubKey("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\KevFLIX",true)){if(key!=null&&previousMarker==null)Registry.CurrentUser.DeleteSubKeyTree("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\KevFLIX",false);}}catch{}
            if(existing){try{WriteProfileState("desktop-update-result.json","{\"version\":\""+Version+"\",\"revision\":\""+BuildInfo.Revision+"\",\"ok\":false}");}catch{}}
            throw;
        } finally {
            instanceLock.Dispose();guardLock.Dispose();
            if(existing||success) {
                string python=Path.Combine(Root,".venv/Scripts/pythonw.exe");if(!File.Exists(python))python=Path.Combine(Root,"runtime/python/pythonw.exe");
                if(File.Exists(python))Run(python,"\""+Path.Combine(Root,Updating?"scripts/run_worker.py":"scripts/open_pc.py")+"\"",false);
            }
        }
    }
    // Small status files in the user's PROFILE (relay/desktop_updates.py reads them there); never in the install dir.
    static void WriteProfileState(string name,string text) {
        string dir=Path.Combine(ProfileRoot(),"state");Directory.CreateDirectory(dir);
        string path=Path.Combine(dir,name), tmp=path+"."+Guid.NewGuid().ToString("N")+".tmp";
        File.WriteAllText(tmp,text);
        if(File.Exists(path))File.Replace(tmp,path,null);else File.Move(tmp,path);
    }
    // "sha256  path" lines (app-files.txt / legacy-shipped.txt) -> owned[path] = sha256
    static void ReadOwned(string text,Dictionary<string,string> owned) {
        foreach(var raw in text.Split('\n')) {
            string line=raw.Trim();
            int space=line.IndexOf("  ");
            if(space!=64)continue;
            string sha=line.Substring(0,64).ToLowerInvariant(), name=line.Substring(66).Replace('\\','/');
            if(name.Length>0&&!name.Contains(".."))owned[name]=sha;
        }
    }
    static string Rel(string path) {
        string prefix=Path.GetFullPath(Root).TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar;
        return Path.GetFullPath(path).Substring(prefix.Length).Replace('\\','/');
    }
    static string Sha256(string path) {
        using(var sha=System.Security.Cryptography.SHA256.Create())using(var stream=File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","").ToLowerInvariant();
    }
    static int RunCapture(string file,string args,out string stdout) {
        var p=Process.Start(new ProcessStartInfo(file,args){WorkingDirectory=Root,UseShellExecute=false,CreateNoWindow=true,
            WindowStyle=ProcessWindowStyle.Hidden,RedirectStandardOutput=true});
        stdout=p.StandardOutput.ReadToEnd();
        p.WaitForExit();return p.ExitCode;
    }
    static void Run(string file,string args,bool wait) {
        var p=Process.Start(new ProcessStartInfo(file,args){WorkingDirectory=Root,UseShellExecute=false,CreateNoWindow=true,WindowStyle=ProcessWindowStyle.Hidden});
        if(wait){p.WaitForExit();if(p.ExitCode!=0)throw new Exception("PC configuration could not finish. No media files were changed.");}
    }
}
