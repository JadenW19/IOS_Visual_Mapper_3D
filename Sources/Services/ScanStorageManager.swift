import Foundation

/// Persists baked 3D Dollhouse scans, finalized equipment rosters, and Wavefront `.obj` exports
/// in the app's local `Documents/SavedScans` directory for offline reloading.
public actor ScanStorageManager {
    private let baseDirectory: URL

    public init(baseDirectory: URL? = nil) {
        if let baseDirectory {
            self.baseDirectory = baseDirectory
        } else {
            let fileManager = FileManager()
            let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
                ?? fileManager.temporaryDirectory
            self.baseDirectory = documents.appendingPathComponent("SavedScans", isDirectory: true)
        }
    }

    /// Saves a baked room scan and its equipment roster to disk, including `.obj` and optional photorealistic texture atlas.
    @discardableResult
    public func saveScan(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = Date(),
        environmentMode: EnvironmentMode = .clinicalLab,
        geometry: BakedRoomGeometry,
        equipment: [TrackedEquipment3D],
        textureAtlasJPEGData: Data? = nil
    ) throws -> SavedRoomScanMetadata {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedName = trimmedName.isEmpty
            ? "Scan \(createdAt.formatted(date: .abbreviated, time: .shortened))"
            : trimmedName

        let metadata = SavedRoomScanMetadata(
            id: id,
            name: resolvedName,
            createdAt: createdAt,
            environmentMode: environmentMode,
            equipment: equipment,
            boundsMin: geometry.boundsMin,
            boundsMax: geometry.boundsMax,
            vertexCount: geometry.vertexCount,
            triangleCount: geometry.triangleCount,
            hasTextureAtlas: textureAtlasJPEGData != nil
        )

        let fileManager = FileManager()
        let scanDirectory = directoryURL(for: id)

        do {
            try fileManager.createDirectory(
                at: scanDirectory,
                withIntermediateDirectories: true
            )

            let jsonEncoder = JSONEncoder()
            jsonEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            jsonEncoder.dateEncodingStrategy = .iso8601
            let metadataData = try jsonEncoder.encode(metadata)
            try metadataData.write(to: metadataURL(in: scanDirectory), options: .atomic)

            let plistEncoder = PropertyListEncoder()
            plistEncoder.outputFormat = .binary
            let geometryData = try plistEncoder.encode(geometry)
            try geometryData.write(to: geometryURL(in: scanDirectory), options: .atomic)

            let objData = RoomModelBaker.buildOBJData(from: geometry)
            try objData.write(to: objURL(in: scanDirectory), options: .atomic)

            let atlasFile = textureAtlasURL(in: scanDirectory)
            if let textureAtlasJPEGData, !textureAtlasJPEGData.isEmpty {
                try textureAtlasJPEGData.write(to: atlasFile, options: .atomic)
            } else if fileManager.fileExists(atPath: atlasFile.path) {
                try? fileManager.removeItem(at: atlasFile)
            }

            return metadata
        } catch let error as VisualMapperError {
            throw error
        } catch {
            throw VisualMapperError.scanStorageFailed(error.localizedDescription)
        }
    }

    /// Lists all saved scans sorted newest-first.
    public func listSavedScans() throws -> [SavedRoomScanMetadata] {
        let fileManager = FileManager()
        guard fileManager.fileExists(atPath: baseDirectory.path) else {
            return []
        }

        do {
            let entries = try fileManager.contentsOfDirectory(
                at: baseDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601

            var results: [SavedRoomScanMetadata] = []
            for entry in entries {
                let metaFile = metadataURL(in: entry)
                guard fileManager.fileExists(atPath: metaFile.path) else {
                    continue
                }
                if let data = try? Data(contentsOf: metaFile),
                   let metadata = try? decoder.decode(SavedRoomScanMetadata.self, from: data) {
                    results.append(metadata)
                }
            }

            return results.sorted { $0.createdAt > $1.createdAt }
        } catch {
            throw VisualMapperError.scanStorageFailed(error.localizedDescription)
        }
    }

    /// Loads a saved scan's metadata and full baked 3D geometry.
    public func loadScan(id: UUID) throws -> BakedRoomScan {
        let fileManager = FileManager()
        let scanDirectory = directoryURL(for: id)
        let metaFile = metadataURL(in: scanDirectory)
        let geomFile = geometryURL(in: scanDirectory)

        guard fileManager.fileExists(atPath: metaFile.path),
              fileManager.fileExists(atPath: geomFile.path)
        else {
            throw VisualMapperError.scanStorageFailed("Saved scan \(id.uuidString) could not be found.")
        }

        do {
            let metaData = try Data(contentsOf: metaFile)
            let jsonDecoder = JSONDecoder()
            jsonDecoder.dateDecodingStrategy = .iso8601
            let metadata = try jsonDecoder.decode(SavedRoomScanMetadata.self, from: metaData)

            let geomData = try Data(contentsOf: geomFile)
            let plistDecoder = PropertyListDecoder()
            let geometry = try plistDecoder.decode(BakedRoomGeometry.self, from: geomData)

            let atlasFile = textureAtlasURL(in: scanDirectory)
            let textureAtlasData = fileManager.fileExists(atPath: atlasFile.path)
                ? try? Data(contentsOf: atlasFile)
                : nil

            return BakedRoomScan(
                metadata: metadata,
                geometry: geometry,
                textureAtlasData: textureAtlasData
            )
        } catch let error as VisualMapperError {
            throw error
        } catch {
            throw VisualMapperError.scanStorageFailed(error.localizedDescription)
        }
    }

    /// Deletes a saved scan bundle from disk.
    public func deleteScan(id: UUID) throws {
        let fileManager = FileManager()
        let scanDirectory = directoryURL(for: id)
        guard fileManager.fileExists(atPath: scanDirectory.path) else {
            return
        }

        do {
            try fileManager.removeItem(at: scanDirectory)
        } catch {
            throw VisualMapperError.scanStorageFailed(error.localizedDescription)
        }
    }

    /// Returns the URL of the exported Wavefront `.obj` file for a saved scan if it exists.
    public func exportedOBJURL(for id: UUID) -> URL? {
        let fileManager = FileManager()
        let url = objURL(in: directoryURL(for: id))
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }

    /// Returns the URL of the baked texture atlas JPEG for a saved scan if it exists.
    public func exportedTextureAtlasURL(for id: UUID) -> URL? {
        let fileManager = FileManager()
        let url = textureAtlasURL(in: directoryURL(for: id))
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - Path Helpers

    private func directoryURL(for id: UUID) -> URL {
        baseDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func metadataURL(in directory: URL) -> URL {
        directory.appendingPathComponent("metadata.json", isDirectory: false)
    }

    private func geometryURL(in directory: URL) -> URL {
        directory.appendingPathComponent("geometry.plist", isDirectory: false)
    }

    private func objURL(in directory: URL) -> URL {
        directory.appendingPathComponent("model.obj", isDirectory: false)
    }

    private func textureAtlasURL(in directory: URL) -> URL {
        directory.appendingPathComponent("texture_atlas.jpg", isDirectory: false)
    }
}
