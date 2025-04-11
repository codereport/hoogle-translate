(ns core
  (:require
   [reagent.dom :as rdom]
   [reagent.core :as r]
   [clojure.string :as str]
   [stylefy.core :as stylefy]
   [styles :as styles]
   [data :as data]
   [imgs :as imgs]
   ["react-social-icons" :refer [SocialIcon]]))

(defonce state (r/atom {:top-padding "250px"
                        :results-table [:tr]
                        :theme :light
                        :hidden-langs #{}
                        :settings-open false
                        :show-expressions false
                        :show-libraries false}))

(defonce debug (r/atom {:info ""}))

(defonce debounce-timer (r/atom nil))


; (defn extract-lang [s]
;   (first (str/split s #"@")))

(defn paren-trim [s]
  (if (str/includes? s "(")
    (let [start (str/index-of s "(")
          end (str/index-of s ")")]
      (subs s (inc start) end))
    s))

(defn normalize-algo [algo]
  (->> algo
       (paren-trim)
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
(defn get-lib  [m] (get m :lib))
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

(declare generate-table perform-search)

(def excel-colors ["#CC99FF" "#99CCFF" "#CCFFCC" ;"#CCFFFF"
                   "#FFFF99" "#FFCC99" "#FF99CC" "white"]) 

(defn get-algorithm-font [lang]
  (case lang
    "APL" "'APL387', monospace"
    "Kap" "'APL387', monospace"
    "BQN" "'BQN386', monospace"
    "TinyAPL" "'TinyAPL386', monospace"
    "Uiua" "'Uiua386', monospace"
    "'JetBrains Mono', monospace"))

(def third-party-libraries
  {"python"  ["RAPIDS cuDF" "pandas" "NumPy" "range-v3" "core.matrix", "more-itertools"]
   "c++"     ["range-v3"]
   "rust"    ["itertools"]
   "clojure" ["core.matrix"]})

(defn maybe-filter-third-party-libraries [coll]
  (filter (fn [item]
            (let [lang (normalize-lang (get-lang item))
                  lib  (get-lib item)]
              (or (:show-libraries @state)
                  (not (and lib (some #(= lib %) (get third-party-libraries lang)))))))
          coll))

(defn maybe-filter-expressions [coll]
  (filter (fn [item]
            (or (:show-expressions @state)
                (not (:expr item))))
          coll))

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

(defn get-logo-filename [lang theme]
  (let [base-filename (get imgs/logo-map lang)
        has-dark-variant (contains? imgs/darkmode-logos lang)]
    (if (and (= theme :dark) has-dark-variant)
      ;; Insert "_darkmode" before the file extension
      (let [dot-pos (str/last-index-of base-filename ".")]
        (str (subs base-filename 0 dot-pos) "_darkmode" (subs base-filename dot-pos)))
      base-filename)))

(defn generate-row [info-map color-index]
  (let [current-theme (@state :theme)
        colors (get styles/theme-colors current-theme)
        text-color (:text colors)
        lang (get info-map :lang)]
    [:tr 
     {:on-click
      (fn [e] 
        (if (.-ctrlKey e)
          ;; If Ctrl key is pressed, add language to hidden-langs
          (do
            (.stopPropagation e)
            (swap! state update :hidden-langs conj lang)
            ;; Re-generate the whole table with updated frequencies
            (let [current-state @state
                  selection (or (:selection current-state) (:search-text current-state))
                  how-to-generate-table (:how-to-generate-table current-state)]
              (swap! state assoc :results-table 
                     (generate-table selection how-to-generate-table))))
          ;; Regular click behavior
          (reset! state (merge @state
                               {:top-padding "20px"
                                :theme current-theme
                                :hidden-langs (:hidden-langs @state)
                                :selection (get info-map :id)
                                :how-to-generate-table :by-algo-id
                                :results-table (generate-table (get info-map :id) :by-algo-id)})))
      ::stylefy/mode {:on-hover {:background-color (:hover colors)}})}
     [:td {:on-click (fn [e]
                       (.stopPropagation e)
                       (reset! state (merge @state
                                            {:top-padding "20px"
                                             :theme current-theme
                                             :selection (get info-map :lang)
                                             :how-to-generate-table :by-lang
                                             :results-table (generate-table (get info-map :lang) :by-lang)})))}
      [:img {:src (str/join ["/media/logos/" (get-logo-filename (get info-map :lang) current-theme)]) 
             :width "40px" 
             :height "40px"
             :style {:object-fit "contain"}}]]
     
     ;; Second cell - language name
     [:td {:style {:padding "12px 30px"
                    :color text-color}
            :on-click (fn [e]
                        (.stopPropagation e)
                        (reset! state (merge @state
                                             {:top-padding "20px"
                                              :theme current-theme
                                              :selection (get info-map :lang)
                                              :how-to-generate-table :by-lang
                                              :results-table (generate-table (get info-map :lang) :by-lang)})))}
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
                                    (#(if (contains? % "") (assoc % "" 0) %))
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

(defn is-table-mode? [input]
  (and (str/includes? input " ")
       (str/includes? input ",")))

(defn parse-table-mode [input]
  (let [parts (str/split input #" " 2)]
    (if (= (count parts) 2)
      {:lang (first parts)
       :algos (map str/trim (str/split (second parts) #","))}
      nil)))

(defn get-algorithm-id [algo]
  (->> data/by-key-map
       (filter-by-algo algo)
       (select-keys data/by-key-map)
       (vals)
       (first)
       (get-id)))

(defn generate-table-mode-row [language-name algo-entries-by-lang algo-ids color-indices]
  (let [current-theme (@state :theme)
        colors (get styles/theme-colors current-theme)
        text-color (:text colors)]
    [:tr 
     {:on-click
      (fn [e] 
        (if (.-ctrlKey e)
          ;; If Ctrl key is pressed, add language to hidden-langs
          (do
            (.stopPropagation e)
            (swap! state update :hidden-langs conj language-name)
            ;; Re-generate the whole table with updated frequencies
            (let [current-state @state
                  selection (or (:selection current-state) (:search-text current-state))
                  how-to-generate-table (:how-to-generate-table current-state)]
              (swap! state assoc :results-table 
                     (generate-table selection how-to-generate-table))))
          ;; Regular click behavior
          (reset! state (merge @state
                               {:top-padding "20px"
                                :theme current-theme
                                :hidden-langs (:hidden-langs @state)
                                :selection language-name
                                :how-to-generate-table :by-lang
                                :results-table (generate-table language-name :by-lang)})))
      ::stylefy/mode {:on-hover {:background-color (:hover colors)}})}
     
     ;; Language logo cell
     [:td {:on-click (fn [e]
                       (.stopPropagation e)
                       (reset! state (merge @state
                                            {:top-padding "20px"
                                             :theme current-theme
                                             :selection language-name
                                             :how-to-generate-table :by-lang
                                             :results-table (generate-table language-name :by-lang)})))}
      [:img {:src (str/join ["/media/logos/" (get-logo-filename language-name current-theme)]) 
             :width "40px" 
             :height "40px"
             :style {:object-fit "contain"}}]]
     
     ;; Language name cell
     [:td {:style {:padding "12px 30px" :color text-color}
           :on-click (fn [e]
                       (.stopPropagation e)
                       (reset! state (merge @state
                                            {:top-padding "20px"
                                             :theme current-theme
                                             :selection language-name
                                             :how-to-generate-table :by-lang
                                             :results-table (generate-table language-name :by-lang)})))}
      language-name]
     
     ;; Algorithm cells - one for each algorithm ID
     (for [[algo-name algo-id color-index] algo-ids]
       (let [matching-entries (filter #(= algo-id (get-id %)) 
                                     (get algo-entries-by-lang language-name []))
             algo-name2 (some-> matching-entries first get-algo)
             color-index (get color-indices (normalize-algo (or algo-name2 "")) 0)]
         (if (seq matching-entries)
           ;; Found algorithm for this language
           [:td {:style {:padding "12px 30px"
                         :font-weight "bold"
                         :background-color (nth excel-colors color-index)}} 
            (format-algorithm-with-fonts (get-algo (first matching-entries)) language-name)]
           
           ;; Algorithm not found for this language
           [:td {:style {:padding "12px 30px"}} "😢"])))]))

(defn generate-table-mode [table-config]
  (let [lang (:lang table-config)
        algo-names (:algos table-config)
        hidden-langs (:hidden-langs @state)
        
        ;; Get the algorithm IDs for each algorithm name
        algo-ids-with-names (map (fn [algo-name]
                                  (let [algo-id (get-algorithm-id algo-name)]
                                    [algo-name algo-id])) 
                                algo-names)
        
        ;; Get entries for all requested algorithms by ID
        all-algo-entries (mapcat (fn [[_ algo-id]] 
                                  (when algo-id
                                    (->> data/by-key-map
                                         (filter-by-algo-id algo-id)
                                         (select-keys data/by-key-map)
                                         (vals)
                                         (maybe-filter-third-party-libraries)
                                         (maybe-filter-expressions))))
                                algo-ids-with-names)
        
        ;; Group entries by language
        algo-entries-by-lang (group-by get-lang all-algo-entries)
        
        ;; Get all distinct languages that have at least one of the requested algorithms
        all-languages (keys algo-entries-by-lang)
        
        ;; Filter out hidden languages
        filtered-languages (remove #(contains? hidden-langs %) all-languages)
        
        ;; Filter entries to only include non-hidden languages
        filtered-entries (filter #(not (contains? hidden-langs (get-lang %))) all-algo-entries)
        
        ;; Count algorithm occurrence frequencies by ID using only visible languages
        algo-freq (->> filtered-entries
                       (map get-algo)
                       (map normalize-algo)
                       (frequencies)
                       (#(if (contains? % "") (assoc % "" 0) %)))
        
        ;; Sort by frequency and assign color indices
        color-indices (->> algo-freq
                           (sort-by second >)
                           (map first)
                           (map-indexed (fn [i name] [name (min i 6)]))
                           (into {}))
        
        ;; Create algo-ids with colors
        algo-ids-with-colors (map (fn [[name id]]
                                   [name id (get color-indices (normalize-algo name))])
                                 algo-ids-with-names)]
    
    ;; (swap! debug assoc :info (str "Color indices: " (pr-str color-indices)))
    
    [:table {:style {:font-family "'JetBrains Mono', monospace"
                    :padding "12px 12px"
                    :font-size "20"
                    :margin-left "auto"
                    :margin-right "auto"
                    :text-align "center"}}
     
     ;; Table body
     [:tbody
      (->> filtered-languages
           ;; Sort by algorithm frequency instead of by language name
           (sort-by (fn [lang-name]
                     (let [entries (get algo-entries-by-lang lang-name [])
                           ;; Count how many algorithms of each color index the language has
                           color-counts (for [i (range 7)]  ;; 0-6 are our color indices
                                          (count (filter (fn [entry]
                                                   (let [algo-name (get-algo entry)
                                                        normalized (normalize-algo algo-name)
                                                        color (get color-indices normalized 7)]  ;; Default to highest value if not found
                                                     (= color i)))
                                                 entries)))
                           ;; Create a vector for sorting: [count-of-color-0, count-of-color-1, ...] 
                           ;; Negative values to sort in descending order
                           sort-key (vec (map #(- %) color-counts))]
                       ;; Append language name at the end for stable sorting
                       (conj sort-key lang-name))))
           (map #(generate-table-mode-row % algo-entries-by-lang algo-ids-with-colors color-indices)))]]))

(defn decide-how [input]
  (cond
    (is-table-mode? input) :table-mode
    (in? (->> imgs/logo-map
              (keys)
              (map str/lower-case))
         (str/lower-case input)) :by-lang
    (str/includes? input "->") :by-sig
    (str/includes? input " ")  :by-algo-id
    :else :by-algo))

(defn generate-table [selection how-to-generate-table]
  (if (= how-to-generate-table :table-mode)
    (generate-table-mode (parse-table-mode selection))
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
          (remove #(contains? (:hidden-langs @state) (get-lang %)))
          (maybe-filter-third-party-libraries)
          (maybe-filter-expressions)
          (choose-colors how-to-generate-table)
          (sort-by last)
          (map (partial apply generate-row)))]))

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
                 ;; Update body styles with new theme
                 (update-body-styles new-theme)
                 ;; If there are search results showing, regenerate the table
                 (if (= top-padding "20px")
                   (let [search-text (:search-text current-state)
                         selection (:selection current-state)
                         how-to-generate-table (:how-to-generate-table current-state)]
                     ;; Force a complete re-render of the table with the new theme
                     (reset! state (merge @state 
                                       {:search-text search-text
                                        :top-padding top-padding
                                        :theme new-theme
                                        :selection selection
                                        :hidden-langs (:hidden-langs current-state)
                                        :settings-open (:settings-open current-state)
                                        :show-expressions (:show-expressions current-state)
                                        :show-libraries (:show-libraries current-state)
                                        :how-to-generate-table how-to-generate-table}))
                     ;; Update the results table after the state has been updated with the new theme
                     (when (and selection how-to-generate-table)
                       (swap! state assoc :results-table
                              (generate-table selection how-to-generate-table))))
                   ;; Otherwise just update the theme
                   (reset! state (assoc current-state :theme new-theme))))}
   (if (= (@state :theme) :light) "🌙 Dark" "☀️ Light")])

(defn settings-toggle []
  [:button
   {:style (merge (styles/theme-toggle-style (@state :theme))
                 {:margin-top "50px"
                  :display "block"
                  :clear "both"})
    :on-click #(swap! state update :settings-open not)}
   "⚙️"])

(defn settings-panel []
  (when (@state :settings-open)
    (let [current-theme (@state :theme)
          colors (get styles/theme-colors current-theme)]
      [:div {:style {:position "absolute"
                     :left "50px"
                     :top "130px"
                     :background-color (:background colors)
                     :border (str "1px solid " (:border colors))
                     :border-radius "5px"
                     :padding "10px"
                     :z-index 100
                     :box-shadow "0 2px 10px rgba(0,0,0,0.2)"}}
       [:div {:style {:margin "5px 0"
               :text-align "left"}}
        [:label {:style {:font-family "'JetBrains Mono', monospace"
                         :color (:text colors)
                         :margin-left "5px"
                         :user-select "none"}}
         [:input {:type "checkbox"
                  :checked (@state :show-expressions)
                  :on-change (fn [_]
                               (swap! state update :show-expressions not)
                               ;; Then immediately refresh table if results are showing
                               (when (= (@state :top-padding) "20px")
                                 (let [selection (or (:selection @state) (:search-text @state))
                                       how-to-generate-table (:how-to-generate-table @state)]
                                   (when (and selection how-to-generate-table)
                                     (swap! state assoc :results-table
                                            (generate-table selection how-to-generate-table))))))}]
         " Show Expressions"]]
       [:div {:style {:margin "5px 0"
               :text-align "left"}}
        [:label {:style {:font-family "'JetBrains Mono', monospace"
                         :color (:text colors)
                         :margin-left "5px"
                         :user-select "none"}}
         [:input {:type "checkbox"
                  :checked (@state :show-libraries)
                  :on-change (fn [_]
                               (swap! state update :show-libraries not)
                               ;; Then immediately refresh table if results are showing
                               (when (= (@state :top-padding) "20px")
                                 (let [selection (or (:selection @state) (:search-text @state))
                                       how-to-generate-table (:how-to-generate-table @state)]
                                   (when (and selection how-to-generate-table)
                                     (swap! state assoc :results-table
                                            (generate-table selection how-to-generate-table))))))}]
         " Show Third Party Libraries"]]])))

(defn perform-search 
  ([search-text current-theme]
   (perform-search search-text current-theme false))
  ([search-text current-theme reset-hidden?]
   (let [how-to-generate-table (decide-how search-text)
         selection (cond
                    (= how-to-generate-table :table-mode) search-text
                    (and (str/includes? search-text " ")
                         (not (str/includes? search-text "->")))
                    (let [lang-algo (str/split search-text #" ")
                          lang (first lang-algo)
                          algo (second lang-algo)
                          ;; Find the matching entry by prefix matching on language and algorithm
                          matching-keys (filter #(and 
                                                 (str/starts-with? % (str lang "@"))
                                                 (str/includes? % (str "@" algo "@")))
                                               (keys data/by-key-map))
                          matching-key (first matching-keys)]
                      (if matching-key
                        (->> matching-key
                             (get data/by-key-map)
                             (get-id))
                        search-text))
                    :else search-text)
         ;; Use parameter to determine whether to reset hidden languages
         existing-hidden-langs (if reset-hidden? #{} (:hidden-langs @state))]
     (swap! state assoc 
            :search-text search-text
            :top-padding "20px"
            :theme current-theme
            :selection selection
            :hidden-langs existing-hidden-langs
            :how-to-generate-table how-to-generate-table
            :results-table (generate-table selection how-to-generate-table)))))

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
     [settings-toggle]
     [settings-panel]
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
           (perform-search (.. e -target -value) current-theme true)
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
