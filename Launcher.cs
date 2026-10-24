using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Launcher
{
    [STAThread]
    static void Main()
    {
        // Get the directory containing ProphitBet.exe
        string directory = AppDomain.CurrentDomain.BaseDirectory;

        // Locate app.bat
        string batchFile = Path.Combine(directory, "app.bat");

        if (!File.Exists(batchFile))
        {
            MessageBox.Show(
                "Cannot find app.bat in:\n" + directory,
                "ProphitBet - Error",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return;
        }

        // Execute the batch file
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = "/c \"\"" + batchFile + "\"\"",
            WorkingDirectory = directory,
            UseShellExecute = false
        };

        try
        {
            Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "ProphitBet - Launch Error",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }
}