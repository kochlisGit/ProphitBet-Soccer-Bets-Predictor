# breeze-dark.tcl --
#
# breeze-dark pixmap theme for the ttk package.
#
#  Copyright (c) 2020 Bartek Jasicki

package require Tk 8.6.0

namespace eval ttk::theme::breeze-dark {

    variable version 0.1
    package provide ttk::theme::breeze-dark $version

    variable colors
    array set colors {
        -bg             "#31363b"
        -fg             "#eff0f1"
        
        -disabledbg     "#4d4d4d"
        -disabledfg     "#7f8c8d"
        
        -selectbg       "#3daee9"
        -selectfg       "white"
        
        -window         "#31363b"
        -focuscolor     "#3daee9"
        -checklight     "#94d0eb"
    }

    proc LoadImages {imgdir} {
        variable I
        foreach file [glob -directory $imgdir *.png] {
            set img [file tail [file rootname $file]]
            set I($img) [image create photo -file $file -format png]
        }
    }

    LoadImages [file join [file dirname [info script]] breeze-dark]

    ttk::style theme create breeze-dark -parent default -settings {
        ttk::style configure . \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -troughcolor $colors(-bg) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -fieldbackground $colors(-window) \
            -font TkDefaultFont \
            -borderwidth 1 \
            -focuscolor $colors(-focuscolor) \
            -highlightcolor $colors(-checklight)

        ttk::style map . -foreground [list disabled $colors(-disabledfg)]

        #
        # Layouts:
        #

        ttk::style layout TButton {
            Button.button -children {
                
                    Button.padding -children {
                        Button.label -side left -expand true
                    }
                
            }
        }

        ttk::style layout Toolbutton {
            Toolbutton.button -children {
                    Toolbutton.padding -children {
                        Toolbutton.label -side left -expand true
                    }
            }
        }

        ttk::style layout Vertical.TScrollbar {
            Vertical.Scrollbar.trough -sticky ns -children {
                Vertical.Scrollbar.thumb -expand true
            }
        }

        ttk::style layout Horizontal.TScrollbar {
            Horizontal.Scrollbar.trough -sticky ew -children {
                Horizontal.Scrollbar.thumb -expand true
            }
        }

        ttk::style layout TMenubutton {
            Menubutton.button -children {
                Menubutton.focus -children {
                    Menubutton.padding -children {
                        Menubutton.indicator -side right
                        Menubutton.label -side right -expand true
                    }
                }
            }
        }
        
        ttk::style layout Item {
            Treeitem.padding -sticky nswe -children {
                Treeitem.indicator -side left -sticky {} Treeitem.image -side left -sticky {} -children {
                    Treeitem.text -side left -sticky {}
                    }
                }
        }
           
        #
        # Elements:
        #

        ttk::style element create Button.button image [list $I(button) \
                pressed     $I(button-active) \
                {active focus}       $I(button-active) \
                active      $I(button-hover) \
                focus       $I(button-focus) \
                disabled    $I(button-insensitive) \
            ] -border 3 -sticky ewns

        ttk::style element create Toolbutton.button image [list $I(button-empty) \
                {active selected !disabled}  $I(button-active) \
                selected            $I(button-toggled) \
                pressed             $I(button-active) \
                {active !disabled}  $I(button-hover) \
            ] -border 3 -sticky news

        ttk::style element create Checkbutton.indicator image [list $I(checkbox-unchecked) \
                {pressed selected}  $I(checkbox-checked-pressed) \
                {active selected}   $I(checkbox-checked-active) \
                {pressed !selected} $I(checkbox-unchecked-pressed) \
                active              $I(checkbox-unchecked-active) \
                selected            $I(checkbox-checked) \
                {disabled selected} $I(checkbox-checked-insensitive) \
                disabled            $I(checkbox-unchecked-insensitive) \
            ] -width 22 -sticky w

        ttk::style element create Radiobutton.indicator image [list $I(radio-unchecked) \
                disabled            $I(radio-unchecked-insensitive) \
                {pressed selected}  $I(radio-checked-pressed) \
                {active selected}   $I(radio-checked-active) \
                {pressed !selected} $I(radio-unchecked-pressed) \
                active              $I(radio-unchecked-active) \
                selected            $I(radio-checked) \
                {disabled selected} $I(radio-checked-insensitive) \
            ] -width 22 -sticky w

            
        ttk::style element create Horizontal.Scrollbar.trough image $I(scrollbar-trough-horiz-active) \
        -border {6 0 6 0} -sticky ew
        ttk::style element create Horizontal.Scrollbar.thumb \
             image [list $I(scrollbar-slider-horiz) \
                        {active !disabled}  $I(scrollbar-slider-horiz-active) \
                        disabled            $I(scrollbar-slider-insens) \
            ] -border {6 0 6 0} -sticky ew

        ttk::style element create Vertical.Scrollbar.trough image $I(scrollbar-trough-vert-active) \
            -border {0 6 0 6} -sticky ns
        ttk::style element create Vertical.Scrollbar.thumb \
            image [list $I(scrollbar-slider-vert) \
                        {active !disabled}  $I(scrollbar-slider-vert-active) \
                        disabled            $I(scrollbar-slider-insens) \
            ] -border {0 6 0 6} -sticky ns

        
        ttk::style element create Horizontal.Scale.trough \
            image [list $I(scrollbar-slider-horiz) disabled $I(scale-trough-horizontal)] \
            -border {8 5 8 5} -padding 0
        ttk::style element create Horizontal.Scale.slider \
            image [list $I(scale-slider) \
                disabled $I(scale-slider-insensitive) \
                pressed $I(scale-slider-pressed)\
                active $I(scale-slider-active) \
                ] \
            -sticky {}
            
            
        ttk::style element create Vertical.Scale.trough \
            image [list $I(scrollbar-slider-vert) disabled $I(scale-trough-vertical)] \
            -border {8 5 8 5} -padding 0
        ttk::style element create Vertical.Scale.slider \
            image [list $I(scale-slider) \
                disabled $I(scale-slider-insensitive) \
                pressed $I(scale-slider-pressed)\
                active $I(scale-slider-active) \
                ] \
            -sticky {}

        ttk::style element create Entry.field \
            image [list $I(entry) \
                        {focus !disabled} $I(entry-focus) \
                        {hover !disabled} $I(entry-active) \
                        disabled $I(entry-insensitive)] \
            -border 3 -padding {6 8} -sticky news

        ttk::style element create Labelframe.border image $I(labelframe) \
            -border 4 -padding 4 -sticky news

        ttk::style element create Menubutton.button \
            image [list $I(button) \
                        pressed  $I(button-active) \
                        active   $I(button-hover) \
                        disabled $I(button-insensitive) \
            ] -sticky news -border 3 -padding {3 2}
        ttk::style element create Menubutton.indicator \
            image [list $I(arrow-down) \
                        active   $I(arrow-down-prelight) \
                        pressed  $I(arrow-down-prelight) \
                        disabled $I(arrow-down-insens) \
            ] -sticky e -width 20

        ttk::style element create Combobox.field \
            image [list $I(entry) \
                {readonly disabled}  $I(button-insensitive) \
                {readonly pressed}   $I(button-hover) \
                {readonly focus hover}     $I(button-active) \
                {readonly focus}     $I(button-focus) \
                {readonly hover}     $I(button-hover) \
                readonly             $I(button) \
                {disabled} $I(entry-insensitive) \
                {focus}    $I(entry-focus) \
                {focus hover}    $I(entry-focus) \
                {hover}    $I(entry-active) \
            ] -border 4 -padding {6 8}
        ttk::style element create Combobox.downarrow \
            image [list $I(arrow-down) \
                        active    $I(arrow-down-prelight) \
                        pressed   $I(arrow-down-prelight) \
                        disabled  $I(arrow-down-insens) \
          ]  -border 4 -sticky {}

        ttk::style element create Spinbox.field \
            image [list $I(entry) focus $I(entry-focus) disabled $I(entry-insensitive) hover $I(entry-active)] \
            -border 4 -padding {6 8} -sticky news
        ttk::style element create Spinbox.uparrow \
            image [list $I(arrow-up-small) \
                        active    $I(arrow-up-small-prelight) \
                        pressed   $I(arrow-up-small-prelight) \
                        disabled  $I(arrow-up-small-insens) \
            ] -border 4 -sticky {}
        ttk::style element create Spinbox.downarrow \
            image [list $I(arrow-down-small) \
                        active    $I(arrow-down-small-prelight) \
                        pressed   $I(arrow-down-small-prelight) \
                        disabled  $I(arrow-down-small-insens) \
          ] -border 4 -sticky {}

       ttk::style element create Notebook.client \
            image $I(notebook-client) -border 1
        ttk::style element create Notebook.tab \
            image [list $I(notebook-tab-top) \
                        selected    $I(notebook-tab-top-active) \
                        active      $I(notebook-tab-top-hover) \
            ] -padding {12 4 12 4} -border 2

            
        # TODO Enhance
        ttk::style element create Horizontal.Progressbar.trough \
            image $I(scrollbar-trough-horiz-active) -border {6 0 6 0} -sticky ew
        ttk::style element create Horizontal.Progressbar.pbar \
            image $I(scrollbar-slider-horiz) -border {6 0 6 0} -sticky ew

        ttk::style element create Vertical.Progressbar.trough \
            image $I(scrollbar-trough-vert-active) -border {0 6 0 6} -sticky ns
        ttk::style element create Vertical.Progressbar.pbar \
            image $I(scrollbar-slider-vert) -border {0 6 0 6} -sticky ns

        ttk::style element create Treeview.field \
            image $I(treeview) -border 1
        ttk::style element create Treeheading.cell \
            image [list $I(notebook-client) \
                active $I(treeheading-prelight)] \
            -border 1 -padding 4 -sticky ewns
        
        # TODO: arrow-* ist at the moment a little bit too big 
        # the small version is too small :-)
        # And at the moment there are no lines as in the Breeze theme
        # And hover, pressed doesn't work
        ttk::style element create Treeitem.indicator \
            image [list $I(arrow-right) \
                user2 $I(empty) \
                user1 $I(arrow-down) \
                ] \
            -width 15 -sticky w
            
        #
        # Settings:
        #

        ttk::style configure TButton -padding {8 4 8 4} -width -10 -anchor center
        ttk::style configure TMenubutton -padding {8 4 4 4}
        ttk::style configure Toolbutton -padding {6 2} -anchor center
        ttk::style configure TCheckbutton -padding 4
        ttk::style configure TRadiobutton -padding 4
        ttk::style configure TSeparator -background $colors(-bg)
        ttk::style configure TEntry -insertcolor $colors(-fg)

        ttk::style map TPanedwindow -background [list hover $colors(-checklight)]
        ttk::style map TCombobox -selectbackground [list \
            !focus         $colors(-window) \
            {readonly hover} $colors(-checklight) \
            {readonly focus} $colors(-focuscolor) \
            ]
            
        ttk::style map TCombobox -selectforeground [list \
            !focus $colors(-fg) \
            {readonly hover} $colors(-fg) \
            {readonly focus} $colors(-selectfg) \
            ]
        
        # Treeview
        ttk::style configure Treeview -background #2f3336
        ttk::style configure Treeview.Item -padding {2 0 0 0}
        ttk::style map Treeview \
            -background [list selected $colors(-selectbg)] \
            -foreground [list selected $colors(-selectfg)]
        
        # Scrollbar 
        bind TScrollbar <MouseWheel> {
           [lindex [%W cget -command] 0] [lindex [%W cget -command] 1] scroll [expr (-1 * (%D / 120))] units
        }
        bind TScrollbar <Button-4> {
           [lindex [%W cget -command] 0] [lindex [%W cget -command] 1] scroll -1 units
        }
        bind TScrollbar <Button-5> {
           [lindex [%W cget -command] 0] [lindex [%W cget -command] 1] scroll 1 units
        }

        # Some defaults for non ttk-widgets so that they fit
        # to the Breeze-dark theme, too
        tk_setPalette background [ttk::style lookup . -background] \
        	foreground [ttk::style lookup . -foreground] \
        	highlightColor [ttk::style lookup . -focuscolor] \
        	selectBackground [ttk::style lookup . -selectbackground] \
        	selectForeground [ttk::style lookup . -selectforeground] \
        	activeBackground [ttk::style lookup . -selectbackground] \
        	activeForeground [ttk::style lookup . -selectforeground]
        option add *font [ttk::style lookup . -font]
    }
}
