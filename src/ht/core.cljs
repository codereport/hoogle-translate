(ns ht.core
  (:require
   [reagent.dom :as rdom]
   [reagent.core :as r]
   [clojure.string :as str]
   [stylefy.core :as stylefy]
   [ht.styles :as styles]
   [ht.data :as data]
   [ht.imgs :as imgs]
   ["react-social-icons" :refer [SocialIcon]]))

(defonce state (r/atom {:top-padding "250px"
                        :results-table [:tr]
                        :theme :light}))

(defonce debug (r/atom {:info ""}))

(defonce debounce-timer (r/atom nil))


; (defn extract-lang [s]
;   (first (str/split s #"@")))

(defn normalize-algo [algo]
  (->> algo
       (re-seq #"[a-zA-Z0-9]")
       (str/join)
       (str/lower-case)))

(defn normalize-lang [lang]
  (->> lang
       (str/join)
       (str/lower-case)))

(defn extract-algo [s] ;; TODO this is confusingly named (returns a list)
  [(-> s
       (str/split #"@")
       (second)
       (normalize-algo)) s])

(defn first-equals [s [a _]]
  (= a s))

(defn filter-by-algo [algo m]
  (map last (filter (partial first-equals (normalize-algo algo)) (map extract-algo (keys m)))))

(defn get-id   [m] (get m :id))
(defn get-algo [m] (get m :algo))
(defn get-lang [m] (get m :lang))
(defn get-sig  [m] (get m :sig))

(defn filter-by-algo-id [id m]
  (map last (filter (partial first-equals id)
                    (map vector (map get-id (vals m)) (keys m)))))

(defn filter-by-algo-id2 [id m]
  (keep (fn [[k {elem-id :id}]] (when (= id elem-id) k)) m))

(defn filter-by-algo-id3 [id m]
  (->> m 
       (map (fn [[k {elem-id :id}]] (when (= id elem-id) k)))
       (remove nil?)))

(defn filter-by-algo-id4 [id m]
  (->> m 
       (filter (fn [[_ {elem-id :id}]] (= id elem-id)))
       (map first)))


;; (keep (fn [[k {elem-id :id}]] (when (= id elemid) k)) m)

(defn filter-by-lang [lang m]
  (map last (filter (partial first-equals (normalize-lang lang))
                    (map vector (map (comp normalize-lang get-lang) (vals m)) (keys m)))))

(defn filter-by-sig [sig m]
  (map last (filter (partial first-equals sig)
                    (map vector (map get-sig (vals m)) (keys m)))))

(def tr-hover-style {::stylefy/mode {:hover {:background-color "purple"}}})

(declare generate-table)

(def excel-colors ["#CC99FF" "#99CCFF" "#CCFFCC" ;"#CCFFFF"
                   "#FFFF99" "#FFCC99" "#FF99CC" "white"]) 

(defn get-algorithm-font [lang]
  (case lang
    "APL" "'APL387', monospace"
    "Kap" "'APL387', monospace"
    "BQN" "'BQN386', monospace"
    "Uiua" "'Uiua386', monospace"
    "'JetBrains Mono', monospace"))

(defn format-algorithm-with-fonts [algo-text lang]
  (let [special-font (get-algorithm-font lang)
        default-font "'JetBrains Mono', monospace"
        parts (str/split algo-text #" " 2)] ; Split on first space only
    (if (> (count parts) 1)
      ;; If there are multiple parts, apply different fonts
      [:span
       [:span {:style {:font-family special-font :font-weight "normal"}} (first parts)]
       [:span {:style {:font-family default-font}} (str " " (second parts))]]
      ;; Otherwise use the special font for the entire text
      [:span {:style {:font-family special-font}} algo-text])))

(defn generate-row [info-map color-index]
  (let [current-theme (@state :theme)
        colors (get styles/theme-colors current-theme)
        text-color (:text colors)
        lang (get info-map :lang)]
    [:tr 
     {:on-click
      (fn [_] ((reset! state {:top-padding "20px"
                              :theme current-theme
                              :selection (get info-map :id)
                              :how-to-generate-table :by-algo-id
                              :results-table (generate-table (get info-map :id) :by-algo-id)})))
      ::stylefy/mode {:on-hover {:background-color (:hover colors)}}}
     [:td {:on-click (fn [e]
                       (.stopPropagation e)
                       (reset! state {:top-padding "20px"
                                     :theme current-theme
                                     :selection (get info-map :lang)
                                     :how-to-generate-table :by-lang
                                     :results-table (generate-table (get info-map :lang) :by-lang)}))}
      [:img {:src (str/join ["/media/logos/" (get imgs/logo-map (get info-map :lang))]) :width "40px" :height "40px"}]]
     [:td {:style {:padding "12px 30px"
                   :color text-color}
           :on-click (fn [e]
                       (.stopPropagation e)
                       (reset! state {:top-padding "20px"
                                     :theme current-theme
                                     :selection (get info-map :lang)
                                     :how-to-generate-table :by-lang
                                     :results-table (generate-table (get info-map :lang) :by-lang)}))}
      (get info-map :lang)]
     [:td {:style {:padding "12px 30px"
                   :font-weight "bold"
                   :background-color (nth excel-colors color-index)}} 
      (format-algorithm-with-fonts (get info-map :algo) lang)]
     [:td {:style {:padding "12px 30px" :color text-color}} (get info-map :lib)]
     [:td {:style {:padding "12px 30px"}} [:a {:href (get info-map :doc)
                                              :style {:color (:primary colors)}} "Doc"]]]))

(defn choose-filter [how-to-generate-table]
  (case how-to-generate-table
    :by-algo    (partial filter-by-algo)
    :by-algo-id (partial filter-by-algo-id)
    :by-lang    (partial filter-by-lang)
    :by-sig     (partial filter-by-sig)))

(defn choose-colors [how-to-generate-table rows-info]
  (case how-to-generate-table
    
    :by-algo-id (map vector rows-info
                     (map (partial
                           get (->> (map get-algo rows-info)
                                    (map normalize-algo)
                                    (frequencies)
                                    (into (vector))
                                    (sort-by last >)
                                    (map-indexed (fn [i [algo _]] [algo (min i 6)]))
                                    (into (hash-map))))
                          (->> (map get-algo rows-info)
                               (map normalize-algo))))
    
    :by-algo    (map vector rows-info
                     (map (partial
                           get (->> (map get-id rows-info)
                                    (frequencies)
                                    (into (vector))
                                    (sort-by last >)
                                    (map-indexed (fn [i [id _]] [id (min i 6)]))
                                    (into (hash-map)))) (map get-id rows-info)))
    
    :by-lang    (map vector rows-info
                     (repeat (count rows-info) 0))
    
    :by-sig    (map vector rows-info
                    (repeat (count rows-info) 0))))

(defn in?
  [coll elm]
  (some #(= elm %) coll))

(defn decide-how [input]
  (cond 
    (in? (->> imgs/logo-map
              (keys)
              (map str/lower-case))
         (str/lower-case input)) :by-lang ;; TODO figure out why this lower-case isn't working
    (str/includes? input "->") :by-sig
    (str/includes? input " ")  :by-algo-id
    :else :by-algo))

(defn generate-table [selection how-to-generate-table]
  [:table {:style {:font-family "'JetBrains Mono', monospace"
                   :padding "12px 12px"
                   :font-size "20" ; this is for the rows
                   :margin-left "auto"
                   :margin-right "auto"
                   :text-align "center"}}

   (->> data/by-key-map
        ((choose-filter how-to-generate-table) selection)
        (select-keys data/by-key-map)
        (vals)
        (choose-colors how-to-generate-table)
        (sort-by last)
        (map (partial apply generate-row)))

  ])

(defn social-icon [props]
  [:> SocialIcon (merge {:style (styles/social-icon-style)}
                       {:onMouseOver (fn [e] 
                                      (-> e .-currentTarget .-style .-transform (set! "scale(1.25)")))
                        :onMouseOut (fn [e] 
                                     (-> e .-currentTarget .-style .-transform (set! "scale(1)")))}
                       props)])

(defn social-links []
  [:div (styles/social-links-container)
   [social-icon {:url "https://bsky.app/profile/codereport.bsky.social"}]
   [social-icon {:url "https://mastodon.social/@code_report" :network "mastodon"}]
   [social-icon {:url "https://www.twitter.com/code_report"}]
   [social-icon {:url "https://www.youtube.com/c/codereport"}]
   [social-icon {:url "https://www.github.com/codereport"}]])

(def footnote {:style {:font-size 12}})

(defn footnotes []
  [:div {:style {:font-size 12 :font-family "'JetBrains Mono', monospace"}}
   [:br]
   [:label "If you would like to contribute a missing language or algorithm, "]
   [:a {:href "https://github.com/codereport/hoogle-translate/blob/main/CONTRIBUTING.md"} [:label "file a PR"]]
   [:label "."]])

(defn save-theme-to-storage [theme]
  (.setItem js/localStorage "ht-theme" (name theme)))

(defn load-theme-from-storage []
  (if-let [saved-theme (.getItem js/localStorage "ht-theme")]
    (keyword saved-theme)
    (:theme @state))) ; default to current state if no saved theme

(defn update-body-styles [theme]
  (let [colors (get styles/theme-colors theme)
        style-element (or (.getElementById js/document "theme-styles")
                          (let [el (.createElement js/document "style")]
                            (set! (.-id el) "theme-styles")
                            (.appendChild (.-head js/document) el)
                            el))]
    (set! (.-innerHTML style-element) 
          (str "body { background-color: " (:background colors) "; margin: 0; padding: 0; }"))))

(defn theme-toggle []
  [:button
   {:style (styles/theme-toggle-style (@state :theme))
    :on-click #(let [new-theme (if (= (@state :theme) :light) :dark :light)
                     current-state @state
                     top-padding (:top-padding current-state)]
                 ;; Save theme to localStorage
                 (save-theme-to-storage new-theme)
                 ;; If there are search results showing, regenerate the table
                 (if (= top-padding "20px")
                   (let [selection (or (:selection current-state) (:search-text current-state))
                         how-to-generate-table (or (:how-to-generate-table current-state) 
                                                  (when selection (decide-how selection)))]
                     (reset! state {:top-padding top-padding
                                    :theme new-theme
                                    :selection selection
                                    :search-text (:search-text current-state)
                                    :how-to-generate-table how-to-generate-table
                                    :results-table (when (and selection how-to-generate-table)
                                                    (generate-table selection how-to-generate-table))}))
                   ;; Otherwise just update the theme
                   (reset! state (assoc current-state :theme new-theme)))
                 (update-body-styles new-theme))}
   (if (= (@state :theme) :light) "🌙 Dark" "☀️ Light")])

(defn perform-search [search-text current-theme]
  (reset! state {:search-text search-text
                 :top-padding "20px"
                 :theme current-theme
                 :results-table (generate-table (if (and (str/includes? search-text " ")
                                                         (not (str/includes? search-text "->")))
                                                  (->> (str/replace search-text " " "@")
                                                       (get data/by-key-map)
                                                       (get-id))
                                                  search-text) 
                                                (decide-how search-text))}))

(defn debounced-search [search-text current-theme delay-ms]
  (when @debounce-timer
    (js/clearTimeout @debounce-timer))
  (reset! debounce-timer 
          (js/setTimeout #(perform-search search-text current-theme) delay-ms)))

(defn app-view []
  (let [current-theme (@state :theme)
        colors (get styles/theme-colors current-theme)]
    [:div {:style (styles/app-container-style current-theme (@state :top-padding))}
     [theme-toggle]
     [:a {:href "https://www.youtube.com/c/codereport"
          :style (styles/logo-link)}
      [:img {:src "/media/code_report_circle.png"
             :style (styles/logo-image)
             :on-mouse-over (fn [e] 
                             (-> e .-target .-style .-transform (set! "scale(1.25)")))
             :on-mouse-out (fn [e] 
                            (-> e .-target .-style .-transform (set! "scale(1)")))}]]
     
     [:label {:style (styles/heading-style current-theme)} "Hoogle Translate"]
     [:br]
     [:label (@debug :info)]
     [:br]
     [:input
      {:spellcheck "false"
       :focus true
       :style (styles/input-style current-theme)
       :on-change
       (fn [e]
         (let [value (.. e -target -value)]
           (when (not= value "")
             (debounced-search value current-theme 300))))
       :on-key-press
       (fn [e]
         (if (= (.-key e) "Enter")
           (perform-search (.. e -target -value) current-theme)
           (.log js/console "Not Enter")))}]
     [:br]
     [:br]

     (@state :results-table)
     
     ;; Only show attribution and social links when no results are showing
     (when (= (@state :top-padding) "250px")
       [:div
        [:label (styles/font 25) "by code_report"]
        [social-links]])
        
     ;; Only show footnotes when results are showing
     (when (not= (@state :top-padding) "250px")
       [footnotes])
     ]))

(defn init-theme []
  (let [saved-theme (load-theme-from-storage)]
    ;; Update state with saved theme if available
    (when (not= saved-theme (:theme @state))
      (swap! state assoc :theme saved-theme))
    (update-body-styles saved-theme))
  nil)

(defn render! []
  (init-theme)
  (rdom/render
   [app-view]
   (js/document.getElementById "app")))
