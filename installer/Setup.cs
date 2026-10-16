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
class Setup : Form {
    const string Product = "KevFLIX", Version = BuildInfo.Version;
    static string Root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", Product);
    static bool Silent, Updating, Startup = true;
    static int WaitPid;
    Label status; Button install; CheckBox startup;
    Setup() {
        Text="KevFLIX Setup " + Version; Width=650; Height=330; StartPosition=FormStartPosition.CenterScreen;
        Font=new Font("Segoe UI",10); FormBorderStyle=FormBorderStyle.FixedDialog; MaximizeBox=false;
        var title=new Label {Text="Install KevFLIX PC",Left=28,Top=25,Width=560,Height=38,Font=new Font("Segoe UI",20)};
        status=new Label {Text="Installs the desktop control panel and private Python runtime.\nYour Plex account and settings stay on this PC.\nMovie processing tools are installed separately from the control panel.\n\nDevelopment build: playback and installation have not been tested.",Left=30,Top=75,Width=575,Height=135};
        startup=new CheckBox {Text="Start KevFLIX with Windows",Checked=true,Left=30,Top=214,Width=350};
        install=new Button {Text="Install",Left=470,Top=240,Width=135};
        bool finished=false;
        install.Click+=async delegate {if(finished){Close();return;}Startup=startup.Checked;install.Enabled=false;startup.Enabled=false;try {await Task.Run(()=>Install());status.Text="Installed. KevFLIX PC is opening.\nConnect Plex or choose movie folders, then pair your TV.";} catch(Exception ex){status.Text="Installation did not complete: "+ex.Message;}finally{finished=true;install.Text="Close";install.Enabled=true;}};
        Controls.AddRange(new Control[]{title,status,startup,install});
    }
    [STAThread] static int Main(string[] args) {
        try {
            for(int i=0;i<args.Length;i++) {
                if(args[i]=="/silent") Silent=true;
                else if(args[i]=="/update") Updating=true;
                else if(args[i]=="/target") Root=Path.GetFullPath(args[++i]);
                else if(args[i]=="/wait-pid") WaitPid=int.Parse(args[++i]);
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
        string previousMarker=existing?File.ReadAllText(marker):null;
        string oldRequirements=File.Exists(Safe("requirements.txt"))?File.ReadAllText(Safe("requirements.txt")):null;
        var changed=new List<string>();var created=new List<string>();
        string backupName="setup-backups/"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N");
        string backup=Safe(backupName);Directory.CreateDirectory(backup);
        bool success=false, replacedEnvironment=false;
        try {
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
            if(!existing || !File.Exists(Safe("relay-config.json")) || new FileInfo(Safe("relay-config.json")).Length==0) {
                Run(Path.Combine(Root,"runtime/python/python.exe"),"\""+Path.Combine(Root,"scripts/install_pc.py")+"\" --startup "+(Startup?"on":"off"),true);
            }
            if(existing && Directory.Exists(Safe(".venv")) && oldRequirements!=File.ReadAllText(Safe("requirements.txt"))) {
                Directory.Move(Safe(".venv"),Safe(backupName+"/previous-venv"));replacedEnvironment=true;
                Run(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),"WindowsPowerShell/v1.0/powershell.exe"),
                    "-NoProfile -ExecutionPolicy Bypass -File \""+Safe("Setup-Worker.ps1")+"\"",true);
            }
            File.WriteAllText(marker,"KevFLIX\n"+Version+"\n"+BuildInfo.Revision+"\n");
            File.WriteAllText(Safe("state/desktop-update-result.json"),"{\"version\":\""+Version+"\",\"revision\":\""+BuildInfo.Revision+"\",\"ok\":true}");
            using(var key=Registry.CurrentUser.CreateSubKey("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\KevFLIX")) {
                key.SetValue("DisplayName",Product);key.SetValue("DisplayVersion",Version);key.SetValue("Publisher",Product);key.SetValue("InstallLocation",Root);
                key.SetValue("UninstallString","\""+Path.Combine(Root,"runtime/python/pythonw.exe")+"\" \""+Path.Combine(Root,"scripts/uninstall_pc.py")+"\"");key.SetValue("NoModify",1);key.SetValue("NoRepair",1);
            }
            string pending=Safe("state/desktop-updates/pending.json");
            if(File.Exists(pending))File.Delete(pending);
            success=true;
        } catch {
            if(replacedEnvironment){if(Directory.Exists(Safe(".venv")))Directory.Move(Safe(".venv"),Safe(backupName+"/failed-venv"));Directory.Move(Safe(backupName+"/previous-venv"),Safe(".venv"));}
            foreach(var name in changed)File.Copy(Path.Combine(backup,name),Safe(name),true);
            foreach(var name in created)if(File.Exists(Safe(name)))File.Delete(Safe(name));
            if(previousMarker!=null)File.WriteAllText(marker,previousMarker);else File.WriteAllText(marker,"KevFLIX\npartial\n");
            if(existing){Directory.CreateDirectory(Safe("state"));File.WriteAllText(Safe("state/desktop-update-result.json"),"{\"version\":\""+Version+"\",\"revision\":\""+BuildInfo.Revision+"\",\"ok\":false}");}
            throw;
        } finally {
            instanceLock.Dispose();
            if(existing||success) {
                string python=Path.Combine(Root,".venv/Scripts/pythonw.exe");if(!File.Exists(python))python=Path.Combine(Root,"runtime/python/pythonw.exe");
                if(File.Exists(python))Run(python,"\""+Path.Combine(Root,Updating?"scripts/run_worker.py":"scripts/open_pc.py")+"\"",false);
            }
        }
    }
    static void Run(string file,string args,bool wait) {
        var p=Process.Start(new ProcessStartInfo(file,args){WorkingDirectory=Root,UseShellExecute=false,CreateNoWindow=true,WindowStyle=ProcessWindowStyle.Hidden});
        if(wait){p.WaitForExit();if(p.ExitCode!=0)throw new Exception("PC configuration could not finish. No media files were changed.");}
    }
}
