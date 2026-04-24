(ns styles)

(defn font [font-size]
  {:style {:font-family "JetBrains Mono, monospace"
           :font-size (str font-size)
           :font-weight "bold"}})

(defn social-icon-style []
  {:height 40 
   :width 40
   :transition "all 0.2s ease-in-out"
   :transform "scale(1)"})

(defn social-links-container []
  {:style {:display "flex" 
           :gap "10px" 
           :justify-content "center" 
           :margin-top "10px"}})

(defn logo-link []
  {:position "absolute"
   :right "30px"
   :top "30px"
   :cursor "pointer"})

(defn logo-image []
  {:height "60px"
   :width "60px"
   :transition "all 0.2s ease-in-out"
   :transform "scale(1)"})

(def theme-colors
  {:light {:background "white"
           :text "black"
           :primary "darkviolet"
           :border "grey"
           :hover "purple"}
   :dark {:background "#121212"
          :text "white"
          :primary "#bb86fc"
          :border "#757575"
          :hover "#bb86fc"}})

(defn theme-toggle-style [theme]
  {:position "absolute"
   :left "30px"
   :top "30px"
   :background "transparent"
   :border (str "2px solid " (get-in theme-colors [theme :border]))
   :border-radius "25px"
   :padding "8px 16px"
   :color (get-in theme-colors [theme :primary])
   :cursor "pointer"
   :font-family "'JetBrains Mono', monospace"
   :font-weight "bold"
   :transition "all 0.2s ease-in-out"})

(defn input-style [theme]
  (let [colors (get theme-colors theme)]
    {:padding "12px 20px"
     :margin "8px 0"
     :width "600px"
     :font-size "30"
     :font-family "'JetBrains Mono', monospace"
     :border-radius "25px"
     :border (str "2px solid " (:border colors))
     :background-color (:background colors)
     :color (:text colors)
     :outline-width "0"
     :text-align "center"
     :spellcheck "false"}))

(defn app-container-style [theme padding]
  (let [colors (get theme-colors theme)]
    {:search-text ""
     :text-align "center"
     :padding padding
     :background-color (:background colors)
     :color (:text colors)
     :min-height "100vh"
     :overflow "auto"}))

(defn heading-style [theme]
  (let [colors (get theme-colors theme)]
    {:color (:primary colors)
     :font-family "'JetBrains Mono', monospace"
     :font-size "50"
     :font-weight "bold"}))
