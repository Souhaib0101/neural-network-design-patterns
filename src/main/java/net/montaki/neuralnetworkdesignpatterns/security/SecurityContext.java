package net.montaki.neuralnetworkdesignpatterns.security;

import java.util.Set;

/** Identites de demonstration, isolees par thread, sans stockage du mot de passe. */
public final class SecurityContext {
    private static final ThreadLocal<Identity> CURRENT = new ThreadLocal<>();

    private SecurityContext() { }

    public static void authenticate(String username, String password) {
        clear();
        if (!"1234".equals(password)) {
            throw new SecurityException("Identifiants incorrects");
        }
        Set<Role> roles = switch (username == null ? "" : username) {
            case "admin" -> Set.of(Role.ADMIN, Role.TRAINER, Role.VIEWER);
            case "trainer" -> Set.of(Role.TRAINER, Role.VIEWER);
            case "viewer" -> Set.of(Role.VIEWER);
            default -> throw new SecurityException("Identifiants incorrects");
        };
        CURRENT.set(new Identity(username, roles));
    }

    public static boolean isAuthenticated() { return CURRENT.get() != null; }

    public static boolean hasRole(Role role) {
        Identity identity = CURRENT.get();
        return identity != null && identity.roles().contains(role);
    }

    public static String username() { return CURRENT.get() == null ? "anonymous" : CURRENT.get().username(); }

    public static void clear() { CURRENT.remove(); }

    private record Identity(String username, Set<Role> roles) { }
}
