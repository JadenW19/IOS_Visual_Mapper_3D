import Foundation

/// Protocol-driven interface for retrieving equipment functional descriptions, clinical guidelines,
/// safety notes, and enterprise HL7/FHIR/UDI asset identifiers.
public protocol EquipmentKnowledgeProviding: Sendable {
    func fetchContext(
        for item: TrackedEquipment3D,
        mode: EnvironmentMode
    ) async throws -> EquipmentIntelligenceProfile
}

/// Offline, zero-cloud knowledge base covering specialized optometry/clinical equipment
/// and standard COCO-80 / everyday laboratory and residential classes.
public struct LocalEquipmentKnowledgeBase: EquipmentKnowledgeProviding, Sendable {
    fileprivate struct RawEntry: Sendable {
        let displayName: String
        let clinicalCategory: String
        let householdCategory: String
        let clinicalSummary: String
        let householdSummary: String
        let clinicalGuidelines: String
        let householdGuidelines: String
        let maintenanceNotes: String
        let fhirCode: String
        let udiPrefix: String
    }

    private let entries: [String: RawEntry]

    public init() {
        self.entries = Self.buildKnowledgeEntries()
    }

    public func fetchContext(
        for item: TrackedEquipment3D,
        mode: EnvironmentMode
    ) async throws -> EquipmentIntelligenceProfile {
        profile(forLabel: item.label, itemID: item.id, mode: mode)
    }

    /// Synchronous lookup helper for deterministic offline profile resolution.
    public func profile(
        forLabel rawLabel: String,
        itemID: UUID = UUID(),
        mode: EnvironmentMode
    ) -> EquipmentIntelligenceProfile {
        let normalized = rawLabel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let entry = entries[normalized]
        let displayName = entry?.displayName ?? rawLabel.capitalized

        let shortSuffix = String(itemID.uuidString.prefix(8)).uppercased()
        let safePrefix = (entry?.udiPrefix ?? "GEN").uppercased()
        let stableSlugHash = abs(normalized.unicodeScalars.reduce(5381) { ($0 &* 33) &+ Int($1.value) }) % 900000 + 100000

        let category: String
        let summary: String
        let guidelines: String
        switch mode {
        case .clinicalLab:
            category = entry?.clinicalCategory ?? "Clinical Spatial Asset"
            summary = entry?.clinicalSummary
                ?? "Spatial equipment anchor calibrated within the Aunty.cloud clinical simulation environment."
            guidelines = entry?.clinicalGuidelines
                ?? "Verify clear physical clearance, sanitize patient-facing contact zones, and log spatial placement."
        case .householdTest:
            category = entry?.householdCategory ?? "Household Validation Asset"
            summary = entry?.householdSummary
                ?? "Recognized environmental object mapped for spatial validation, bounding extent, and camera focus testing."
            guidelines = entry?.householdGuidelines
                ?? "Inspect object clearance from adjacent surfaces and validate 3D centroid alignment in Dollhouse view."
        }

        let maintenance = entry?.maintenanceNotes
            ?? "Inspect physical alignment, clean optical/touch surfaces per standard protocol, and validate calibration periodically."

        let assetTagID = "AC-\(safePrefix)-\(shortSuffix)"
        let fhirCode = entry?.fhirCode ?? "http://snomed.info/sct|\(stableSlugHash)"
        let udiBarcode = "(01)0084\(safePrefix)\(shortSuffix)(21)\(shortSuffix)"

        return EquipmentIntelligenceProfile(
            canonicalLabel: normalized,
            displayName: displayName,
            category: category,
            clinicalOrFunctionalSummary: summary,
            operationalGuidelines: guidelines,
            maintenanceOrSafetyNotes: maintenance,
            assetTagID: assetTagID,
            fhirDeviceCode: fhirCode,
            udibaBarcode: udiBarcode,
            source: .offlineKnowledgeBase
        )
    }

    /// Number of explicitly mapped class keys in the built-in offline dictionary.
    public var entryCount: Int {
        entries.count
    }

    private static func buildKnowledgeEntries() -> [String: RawEntry] {
        var map: [String: RawEntry] = [:]

        func add(
            _ keys: [String],
            displayName: String,
            clinicalCategory: String,
            householdCategory: String = "Residential Reference Object",
            clinicalSummary: String,
            householdSummary: String,
            clinicalGuidelines: String,
            householdGuidelines: String,
            maintenanceNotes: String,
            fhirCode: String,
            udiPrefix: String
        ) {
            let entry = RawEntry(
                displayName: displayName,
                clinicalCategory: clinicalCategory,
                householdCategory: householdCategory,
                clinicalSummary: clinicalSummary,
                householdSummary: householdSummary,
                clinicalGuidelines: clinicalGuidelines,
                householdGuidelines: householdGuidelines,
                maintenanceNotes: maintenanceNotes,
                fhirCode: fhirCode,
                udiPrefix: udiPrefix
            )
            for key in keys {
                map[key] = entry
            }
        }

        // Specialized Optometry & Medical Simulation Equipment
        add(
            ["phoropter"],
            displayName: "Manual / Digital Phoropter",
            clinicalCategory: "Ophthalmic Diagnostic Instrument",
            clinicalSummary: "Binocular subjective refraction head housing spherical, cylindrical, and prism lens wheels to measure refractive error and binocular vision balance.",
            householdSummary: "Optical clinical instrument proxy for fine-grained spatial anchor placement.",
            clinicalGuidelines: "Align interpupillary distance (PD) and vertex distance before dialing sphere/cylinder power during refraction.",
            householdGuidelines: "Verify suspended arm clearance above exam seating.",
            maintenanceNotes: "Disinfect forehead rest and cheek shields between every patient; clean aperture lenses strictly with optical lens wipes.",
            fhirCode: "http://snomed.info/sct|467861001",
            udiPrefix: "PHR"
        )
        add(
            ["slit lamp"],
            displayName: "Slit Lamp Biomicroscope",
            clinicalCategory: "Ophthalmic Diagnostic Instrument",
            clinicalSummary: "High-intensity focused light source paired with a stereoscopic microscope for anterior and posterior segment ocular examination.",
            householdSummary: "Optical diagnostic workstation mapped as a primary benchtop medical device.",
            clinicalGuidelines: "Lock joystick base when idle, sanitize chin rest and forehead band, and adjust illumination rheostat prior to patient alignment.",
            householdGuidelines: "Verify table height clearance and stable base anchoring.",
            maintenanceNotes: "Use dust cover when inactive; verify illumination bulb calibration and smooth X-Y-Z joystick travel weekly.",
            fhirCode: "http://snomed.info/sct|464042006",
            udiPrefix: "SLT"
        )
        add(
            ["autorefractor"],
            displayName: "Autorefractor / Keratometer",
            clinicalCategory: "Ophthalmic Screening Device",
            clinicalSummary: "Automated infrared objective refraction and corneal curvature analyzer used for rapid baseline refractive screening.",
            householdSummary: "Tabletop optical device mapped for baseline exam-room layout verification.",
            clinicalGuidelines: "Instruct patient to fixate on the internal target; sanitize forehead bar and chin cup after each measurement.",
            householdGuidelines: "Maintain unobstructed approach space at the exam table edge.",
            maintenanceNotes: "Run daily model-eye verification test and clean objective IR window monthly.",
            fhirCode: "http://snomed.info/sct|466801003",
            udiPrefix: "AUR"
        )
        add(
            ["trial lens set"],
            displayName: "Trial Lens Frame & Case",
            clinicalCategory: "Refraction Accessory Kit",
            clinicalSummary: "Calibrated spherical, cylindrical, and occluder lens assortment for free-space and low-vision manual refraction.",
            householdSummary: "Compact benchtop case tracked under tabletop 3D NMS radius.",
            clinicalGuidelines: "Return each lens to its labelled slot immediately to preserve diopter ordering.",
            householdGuidelines: "Verify storage inside drawer or tray surface.",
            maintenanceNotes: "Clean glass trial lenses with non-abrasive optical solution; check rim engraving legibility.",
            fhirCode: "http://snomed.info/sct|463218007",
            udiPrefix: "TLS"
        )
        add(
            ["lens drawer"],
            displayName: "Ophthalmic Lens & Frame Drawer",
            clinicalCategory: "Clinical Storage Unit",
            clinicalSummary: "Organized clinical cabinetry storing trial lenses, diagnostic drops, prism bars, and disposable exam supplies.",
            householdSummary: "Multi-compartment storage cabinet mapped for spatial clearance.",
            clinicalGuidelines: "Keep drawer paths unobstructed by stools or rolling instrument stands.",
            householdGuidelines: "Verify pull clearance in front of workstation.",
            maintenanceNotes: "Audit sterile consumables monthly and keep drawers closed when unattended.",
            fhirCode: "http://snomed.info/sct|469412008",
            udiPrefix: "LDR"
        )
        add(
            ["optometry chair", "chair", "armchair", "rolling stool"],
            displayName: "Clinical / Examination Chair",
            clinicalCategory: "Patient Positioning & Practitioner Seating",
            householdCategory: "Ergonomic Seating",
            clinicalSummary: "Height-adjustable patient exam chair or practitioner pneumatic stool anchoring the primary clinical workstation zone.",
            householdSummary: "Residential chair mapped to evaluate room circulation and desk ergonomics.",
            clinicalGuidelines: "Verify hydraulic brake lock before patient transfer and ensure clear 0.9 m wheelchair turning clearance.",
            householdGuidelines: "Check clearance behind desk/table work zones.",
            maintenanceNotes: "Sanitize upholstery with hospital-grade wipes; inspect casters and hydraulic lift cylinders monthly.",
            fhirCode: "http://snomed.info/sct|466248009",
            udiPrefix: "CHR"
        )
        add(
            ["sharps bin"],
            displayName: "Biohazard Sharps Container",
            clinicalCategory: "Infection Control & Safety",
            clinicalSummary: "Puncture-resistant medical waste container for safe disposal of lancets, needles, and contaminated single-use sharps.",
            householdSummary: "Safety disposal container proxy.",
            clinicalGuidelines: "Keep upright and accessible near point-of-care without obstructing walkways; seal permanently at the 3/4 fill line.",
            householdGuidelines: "Place securely out of reach along lab perimeter.",
            maintenanceNotes: "Never overfill past indicated fill line; replace when 75% full per OSHA biohazard protocol.",
            fhirCode: "http://snomed.info/sct|706724001",
            udiPrefix: "SHP"
        )
        add(
            ["defibrillator"],
            displayName: "Automated External Defibrillator (AED)",
            clinicalCategory: "Emergency Resuscitation Equipment",
            clinicalSummary: "Portable cardiac emergency defibrillator delivering automated rhythm analysis and biphasic shock therapy.",
            householdSummary: "High-priority emergency life-safety device.",
            clinicalGuidelines: "Maintain zero obstruction in front of AED mount; verify audible readiness status indicator daily.",
            householdGuidelines: "Ensure clear line-of-sight access.",
            maintenanceNotes: "Check battery and electrode pad expiration dates monthly per emergency readiness log.",
            fhirCode: "http://snomed.info/sct|469204003",
            udiPrefix: "AED"
        )
        add(
            ["examination light"],
            displayName: "Articulated Examination Light",
            clinicalCategory: "Clinical Illumination",
            clinicalSummary: "Color-corrected high-CRI articulated exam luminaire providing shadow-free illumination during procedures.",
            householdSummary: "Task/floor lighting fixture mapped in 3D space.",
            clinicalGuidelines: "Position lamp handle away from sterile field collisions and disinfect handle sheath after procedures.",
            householdGuidelines: "Check overhead swing radius against adjacent walls.",
            maintenanceNotes: "Inspect tension springs and LED driver thermal vents quarterly.",
            fhirCode: "http://snomed.info/sct|465514006",
            udiPrefix: "LGT"
        )

        // Core Spatial / Planar / Workstation Classes
        add(
            ["monitor", "tvmonitor"],
            displayName: "Clinical / Workstation Monitor",
            clinicalCategory: "Digital Display & EHR Terminal",
            householdCategory: "Desktop Display Monitor",
            clinicalSummary: "High-resolution diagnostic and EHR workstation display used for patient chart review, OCT imaging, and digital acuity charts.",
            householdSummary: "Desktop monitor disambiguated via physical width (< 0.75 m) or desk/keyboard context.",
            clinicalGuidelines: "Angle screen away from direct waiting-room sightlines to uphold HIPAA patient privacy.",
            householdGuidelines: "Verify comfortable viewing distance (~0.6–0.8 m) from desk chair.",
            maintenanceNotes: "Clean display anti-glare coating with alcohol-free microfiber cloth; check DisplayPort/power cable strain relief.",
            fhirCode: "http://snomed.info/sct|462738004",
            udiPrefix: "MON"
        )
        add(
            ["television", "tv"],
            displayName: "Wall / Lounge Television Display",
            clinicalCategory: "Patient Education & Waiting Room Display",
            householdCategory: "Large-Format Television",
            clinicalSummary: "Large-format display (> 0.85 m width) positioned for waiting-room triage queues, digital signage, or calibrated visual acuity projection.",
            householdSummary: "Wide-format display (> 0.85 m) separated from desktop context by 3D volumetric NMS.",
            clinicalGuidelines: "Confirm wall bracket seismic anchor torque and avoid glare reflections across exam lanes.",
            householdGuidelines: "Maintain minimum 2.0 m comfortable viewing setback and clear mounting wall clearance.",
            maintenanceNotes: "Inspect VESA mount fasteners semi-annually and keep rear ventilation slots dust-free.",
            fhirCode: "http://snomed.info/sct|706767009",
            udiPrefix: "TVD"
        )
        add(
            ["whiteboard"],
            displayName: "Clinical Schedule / Teaching Whiteboard",
            clinicalCategory: "Team Communication Surface",
            householdCategory: "Wall Whiteboard / Planning Surface",
            clinicalSummary: "Wall-mounted dry-erase board displaying room turnover status, simulation objectives, or clinical huddle checklists.",
            householdSummary: "Large wall-mounted planar planning surface.",
            clinicalGuidelines: "Avoid writing identifiable PHI on shared public boards without privacy coverage.",
            householdGuidelines: "Ensure clear standing corridor in front of board.",
            maintenanceNotes: "Clean weekly with whiteboard conditioner and replace depleted dry-erase markers.",
            fhirCode: "http://snomed.info/sct|469901001",
            udiPrefix: "WBD"
        )
        add(
            ["desk"],
            displayName: "Clinical Workstation Desk",
            clinicalCategory: "Clinician Documentation Station",
            householdCategory: "Computer Desk / Study Workstation",
            clinicalSummary: "Ergonomic work surface supporting EHR terminals, diagnostic laptops, barcode scanners, and documentation peripherals.",
            householdSummary: "Tabletop reclassified as a desk when paired with monitor, laptop, or keyboard within 0.6 m.",
            clinicalGuidelines: "Keep pathways around clinician knee space clear; manage power strips off the floor.",
            householdGuidelines: "Verify desktop clearance for peripherals and ergonomic seating height.",
            maintenanceNotes: "Sanitize work surface daily and inspect cable routing monthly.",
            fhirCode: "http://snomed.info/sct|468122003",
            udiPrefix: "DSK"
        )
        add(
            ["table", "dining table"],
            displayName: "Utility / Instrument Table",
            clinicalCategory: "Clinical Surface & Staging Area",
            householdCategory: "Table / Multi-Use Surface",
            clinicalSummary: "Flat horizontal staging surface used for instrument placement, tray preparation, or patient consultation.",
            householdSummary: "Standalone horizontal table surface verified free of primary workstation electronics.",
            clinicalGuidelines: "Distinguish clean prep zone from soiled instrument staging zone per infection control workflow.",
            householdGuidelines: "Verify perimeter walk clearance around table corners.",
            maintenanceNotes: "Disinfect top surface between sessions and check leveling feet.",
            fhirCode: "http://snomed.info/sct|465891005",
            udiPrefix: "TBL"
        )
        add(
            ["laptop"],
            displayName: "Mobile Clinical Laptop / Workstation",
            clinicalCategory: "Portable EHR / Diagnostic Compute",
            householdCategory: "Portable Laptop Computer",
            clinicalSummary: "Portable compute terminal running clinical charting, telemetry logging, or portable diagnostic interfaces.",
            householdSummary: "Compact computer unit contributing desktop context for workstation disambiguation.",
            clinicalGuidelines: "Lock screen automatically after 2 minutes of inactivity when near patient areas.",
            householdGuidelines: "Ensure stable desk support and adequate thermal airflow underneath chassis.",
            maintenanceNotes: "Disinfect keyboard gently with electronic-safe wipes; keep battery firmware updated.",
            fhirCode: "http://snomed.info/sct|467119002",
            udiPrefix: "LPT"
        )
        add(
            ["keyboard", "mouse"],
            displayName: "Workstation Input Peripheral",
            clinicalCategory: "EHR Human Interface Device",
            householdCategory: "Desk Input Peripheral",
            clinicalSummary: "Primary input hardware anchoring clinician charting workstations and assisting 3D spatial desk recognition.",
            householdSummary: "Input accessory tracked under 0.25 m tabletop NMS radius.",
            clinicalGuidelines: "Sanitize high-touch keys and mouse body between patient exams.",
            householdGuidelines: "Keep aligned ergonomically in front of monitor.",
            maintenanceNotes: "Inspect silicone covers or wireless batteries monthly.",
            fhirCode: "http://snomed.info/sct|466891008",
            udiPrefix: "HID"
        )

        // Common Laboratory / Household / COCO Classes
        add(
            ["bottle", "cup", "mug", "wine glass", "bowl"],
            displayName: "Reagent / Beverage Vessel",
            clinicalCategory: "Lab Solution / Hydration Container",
            householdCategory: "Tabletop Drinkware / Container",
            clinicalSummary: "Small tabletop container (solution bottle or drinkware); distinguish lab saline/cleaning solutions from staff beverages in clinical zones.",
            householdSummary: "Small tabletop vessel tracked via 0.25 m fine spatial NMS.",
            clinicalGuidelines: "Never keep open personal drinkware inside designated sterile prep or specimen handling zones.",
            householdGuidelines: "Verify stable placement away from electronics/keyboards.",
            maintenanceNotes: "Inspect reagent expiration labels or wash daily.",
            fhirCode: "http://snomed.info/sct|706689003",
            udiPrefix: "VSL"
        )
        add(
            ["cabinet", "filing cabinet", "refrigerator", "shelf"],
            displayName: "Storage Cabinet / Cold Chain Unit",
            clinicalCategory: "Clinical Storage & Supply Unit",
            householdCategory: "Vertical Storage / Appliance",
            clinicalSummary: "Enclosed storage unit housing consumables, patient records, or temperature-regulated clinical drops.",
            householdSummary: "Large vertical furniture/appliance volume.",
            clinicalGuidelines: "Verify door swing arc does not collide with patient chairs or doorway egress.",
            householdGuidelines: "Keep front access path clear.",
            maintenanceNotes: "Log temperature checks if used for cold-chain ophthalmic reagents.",
            fhirCode: "http://snomed.info/sct|467302001",
            udiPrefix: "CAB"
        )
        add(
            ["bed", "couch", "sofa", "bench"],
            displayName: "Examination Couch / Lounge Seating",
            clinicalCategory: "Patient Rest / Triage Seating",
            householdCategory: "Living / Resting Furniture",
            clinicalSummary: "Extended horizontal seating or examination bench supporting waiting patients or supine checks.",
            householdSummary: "Large furniture piece anchoring room spatial layout.",
            clinicalGuidelines: "Use disposable paper barrier sheet when used for patient screening.",
            householdGuidelines: "Ensure clear circulation paths around edges.",
            maintenanceNotes: "Inspect seams and clean fabric/vinyl surfaces regularly.",
            fhirCode: "http://snomed.info/sct|466418003",
            udiPrefix: "BDL"
        )
        add(
            ["sink"],
            displayName: "Clinical Handwashing Sink",
            clinicalCategory: "Hand Hygiene Station",
            householdCategory: "Plumbing Fixture / Sink",
            clinicalSummary: "Dedicated hygiene sink supporting WHO 5-moments handwashing compliance before and after patient contact.",
            householdSummary: "Fixed wet-zone plumbing fixture.",
            clinicalGuidelines: "Keep soap dispenser, paper towels, and alcohol rub replenished adjacent to basin.",
            householdGuidelines: "Keep surrounding counter dry and clear of electrical cords.",
            maintenanceNotes: "Flush infrequently used taps weekly and inspect aerator hygiene.",
            fhirCode: "http://snomed.info/sct|465112004",
            udiPrefix: "SNK"
        )
        add(
            ["door"],
            displayName: "Room Access Doorway",
            clinicalCategory: "Architectural Egress & Privacy Barrier",
            householdCategory: "Door / Room Portal",
            clinicalSummary: "Primary room access portal governing acoustic confidentiality, wheelchair clearance, and fire egress.",
            householdSummary: "Room boundary portal.",
            clinicalGuidelines: "Keep minimum 0.90 m door-swing clearance completely unobstructed inside and outside.",
            householdGuidelines: "Ensure hinges and latch swing freely.",
            maintenanceNotes: "Sanitize push plates and lever handles hourly in high-traffic clinics.",
            fhirCode: "http://snomed.info/sct|224715009",
            udiPrefix: "DOR"
        )
        add(
            ["telephone", "cell phone", "tablet"],
            displayName: "Mobile / Desk Communication Unit",
            clinicalCategory: "Clinical Paging & Triage Device",
            householdCategory: "Mobile Phone / Tablet",
            clinicalSummary: "Point-of-care communication device for reception coordination or mobile patient check-in.",
            householdSummary: "Handheld communication device tracked on tabletops.",
            clinicalGuidelines: "Disinfect glass screen after handoffs and disable public speakerphone near patient privacy zones.",
            householdGuidelines: "Keep charger cords organized off walkways.",
            maintenanceNotes: "Clean with 70% isopropyl wipe safe for oleophobic coatings.",
            fhirCode: "http://snomed.info/sct|468314007",
            udiPrefix: "COM"
        )

        let extraClasses: [(label: String, display: String, cat: String, summary: String)] = [
            ("person", "Clinical Staff / Subject", "Human Presence", "Detected person inside scanned workspace; excluded from static fixture inventories."),
            ("backpack", "Personal / Emergency Pack", "Portable Bag", "Portable gear bag or emergency go-kit staged on floor or chair."),
            ("handbag", "Personal Handbag", "Visitor / Staff Item", "Personal bag placed near seating area."),
            ("suitcase", "Transport Case", "Equipment Transit Case", "Hard or soft transit case for portable diagnostic gear."),
            ("umbrella", "Umbrella", "Personal Weather Item", "Folded or upright umbrella stored near entryway."),
            ("book", "Clinical Reference / Manual", "Documentation", "Printed clinical manual, binder, or diagnostic reference book."),
            ("clock", "Wall / Desk Timer", "Timekeeping", "Reference timepiece aiding timed clinical tests and exam pacing."),
            ("vase", "Decorative Vase", "Environmental Decor", "Decorative tabletop/ledge vessel."),
            ("potted plant", "Indoor Plant", "Environmental Decor", "Waiting room or office greenery fixture."),
            ("scissors", "Utility / Bandage Scissors", "Instrument / Tool", "Cutting instrument; store sheathed away from patient reach."),
            ("pen", "Charting Pen", "Stationery", "Manual writing instrument on workstation."),
            ("mousepad", "Ergonomic Mousepad", "Desk Accessory", "Workstation pointing surface."),
            ("pipette", "Precision Pipette", "Laboratory Instrument", "Calibrated volumetric liquid dispenser for diagnostic assays."),
            ("thermometer", "Clinical Thermometer", "Vital Signs Device", "Infrared or contact temperature screening instrument."),
            ("barcode scanner", "UDI / Specimen Barcode Scanner", "Asset & EHR Input", "Optical scanner reading GS1 UDI barcodes and patient wristbands."),
            ("sample tray", "Diagnostic Sample Tray", "Laboratory Staging", "Compartmentalized tray for organizing vials or lenses."),
            ("gloves box", "Nitrile Exam Gloves Dispenser", "PPE Infection Control", "Dispenser box for disposable examination gloves."),
            ("wheelchair", "Patient Mobility Wheelchair", "Mobility Aid", "Patient transport chair requiring 1.5 m turning circle clearance."),
            ("radiator", "HVAC Radiator / Heater", "Facility Infrastructure", "Perimeter heating fixture; keep flammable supplies clear."),
            ("microwave", "Lab / Breakroom Microwave", "Appliance", "Thermal heating appliance."),
            ("oven", "Thermal Curing / Kitchen Oven", "Appliance", "High-temperature enclosure; keep clearance around front door."),
            ("toaster", "Countertop Toaster", "Kitchen Appliance", "Heating appliance mapped in residential validation scans."),
            ("remote", "AV / Projector Remote", "Control Accessory", "Infrared/Bluetooth handheld remote control."),
            ("teddy bear", "Pediatric Comfort Toy", "Pediatric Exam Aid", "Fixation target or comfort object in pediatric vision lanes."),
            ("toothbrush", "Hygiene Brush", "Personal Care", "Small personal hygiene item."),
            ("hair drier", "Hand Dryer / Heat Gun", "Utility Tool", "Warm-air blower tool.")
        ]

        for item in extraClasses where map[item.label] == nil {
            map[item.label] = RawEntry(
                displayName: item.display,
                clinicalCategory: item.cat,
                householdCategory: item.cat,
                clinicalSummary: item.summary,
                householdSummary: item.summary,
                clinicalGuidelines: "Verify safe clearance and log physical spatial coordinate inside the Aunty.cloud model.",
                householdGuidelines: "Validate 3D centroid and physical clearance around adjacent objects.",
                maintenanceNotes: "Perform visual inspection and clean according to facility standards.",
                fhirCode: "http://hl7.org/fhir/device-type|\(item.label.replacingOccurrences(of: " ", with: "-"))",
                udiPrefix: String(item.label.prefix(3)).uppercased()
            )
        }

        let remainingCoco = [
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "tie", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "fork", "knife",
            "spoon", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "toilet", "toilet paper"
        ]
        for label in remainingCoco where map[label] == nil {
            map[label] = RawEntry(
                displayName: label.capitalized,
                clinicalCategory: "General Vision Class",
                householdCategory: "General COCO-80 Class",
                clinicalSummary: "General YOLO-World vocabulary class '\(label)' mapped during spatial room scanning.",
                householdSummary: "Everyday validation object '\(label)' recognized by the on-device YOLO-World model.",
                clinicalGuidelines: "Confirm whether item is a calibration prop, simulation aid, or extraneous clutter in the exam bay.",
                householdGuidelines: "Inspect 3D bounding anchor alignment relative to floor and tabletop planes.",
                maintenanceNotes: "Clear non-clinical props from diagnostic workstations before live patient sessions.",
                fhirCode: "http://hl7.org/fhir/device-type|\(label.replacingOccurrences(of: " ", with: "-"))",
                udiPrefix: "YLO"
            )
        }

        return map
    }
}

/// Async network/LLM extension hook for Aunty.cloud enterprise intelligence & hospital EHR APIs.
/// Designed for graceful failure when running completely offline so `HybridEquipmentKnowledgeService`
/// falls back seamlessly to `LocalEquipmentKnowledgeBase`.
public struct LiveCloudKnowledgeProvider: EquipmentKnowledgeProviding, Sendable {
    public let endpointURL: URL?

    public init(endpointURL: URL? = nil) {
        self.endpointURL = endpointURL
    }

    public func fetchContext(
        for item: TrackedEquipment3D,
        mode: EnvironmentMode
    ) async throws -> EquipmentIntelligenceProfile {
        guard let endpointURL else {
            throw VisualMapperError.scanStorageFailed("Offline mode: no Aunty.cloud remote endpoint configured.")
        }

        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 3.5
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        struct QueryPayload: Encodable {
            let label: String
            let mode: String
            let itemID: String
        }
        request.httpBody = try JSONEncoder().encode(
            QueryPayload(label: item.label, mode: mode.rawValue, itemID: item.id.uuidString)
        )

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw VisualMapperError.scanStorageFailed("Aunty.cloud API returned an unexpected status code.")
        }

        return try JSONDecoder().decode(EquipmentIntelligenceProfile.self, from: data)
    }
}

/// Composite knowledge provider that queries remote clinical/LLM APIs when configured
/// and online, automatically falling back to the built-in offline YOLO-80 + clinical dictionary.
public struct HybridEquipmentKnowledgeService: EquipmentKnowledgeProviding, Sendable {
    private let cloudProvider: LiveCloudKnowledgeProvider
    private let offlineProvider: LocalEquipmentKnowledgeBase

    public init(
        cloudEndpointURL: URL? = nil,
        offlineProvider: LocalEquipmentKnowledgeBase = LocalEquipmentKnowledgeBase()
    ) {
        self.cloudProvider = LiveCloudKnowledgeProvider(endpointURL: cloudEndpointURL)
        self.offlineProvider = offlineProvider
    }

    public func fetchContext(
        for item: TrackedEquipment3D,
        mode: EnvironmentMode
    ) async throws -> EquipmentIntelligenceProfile {
        if let remoteProfile = try? await cloudProvider.fetchContext(for: item, mode: mode) {
            return remoteProfile
        }
        return try await offlineProvider.fetchContext(for: item, mode: mode)
    }
}
