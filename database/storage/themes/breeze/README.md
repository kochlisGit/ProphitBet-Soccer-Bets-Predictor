## General information

These are Tk version of KDE Breeze and Breeze Dark themes.

* Breeze version was created by MaxPerl:

  https://github.com/MaxPerl/ttk-Breeze

    Changes:

    * Names of directories
    * Small bugfixes
    * Changed font to the default Tk font
    * Added possibility to move scrollbars with mouse wheel

* Breeze Dark version which was created from the theme above by me

## Installation

### Linux
1. Of course, download one or both themes :)
2. If you want to use both, set your environment variable *TCLLIBPATH* to
   directory where file *pkgIndex.tcl* **outside** both themes is. For example,
   if you put both themes in */home/user/themes* directory, *TCLLIBPATH* must
   be set on `/home/user/themes`.
3. If you want to use only one of themes: set your environment variable
   *TCLLIBPATH* to theme directory. For example, if you want to use only Breeze
   theme and you put it in */home/user/.themes/* directory, *TCLLIBPATH* must
   be set on `/home/user/.themes/breeze`.
4. Edit (or create) your *.Xresources* file and add line:
   * If you want to use Breeze theme: `*TkTheme:breeze`
   * If you want to use Breeze dark theme: `*TkTheme:breeze-dark`
5. Reload your Xorg configuration with command: `xrdb -merge ~/.Xresources`
6. Profit :)

----

That's all for now, as usual, probably I forgot about something important ;)

Bartek thindil Jasicki
